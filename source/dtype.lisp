
(in-package :cl-waffe)

(defparameter *dtypes* `(:float :double)) ; To Add: :half
(defparameter *dtype-prefixes* `(-f -d))
(defparameter *dtype-cl-names* `(single-float double-float))

(defun dtype-p (dtype)
  (if (find dtype *dtypes*)
      dtype
      (error "Invaild dtype ~a. Dtype must be chosen by following: ~a"
	     dtype
	     *dtypes*)))

(defmacro with-dtype (dtype &body body)
  `(let ((mgl-mat:*DEFAULT-MAT-CTYPE* ,(dtype-p dtype)))
     ,@body))

(defun map-tree (fn tree)
  (let ((tree (funcall fn tree)))
    (if (listp tree)
        (mapcar (lambda (subtree)
                  (map-tree fn subtree))
                tree)
        tree)))

(defun replace-lisp-code-with-dtype (body type-var dtype)
  (map-tree #'(lambda (code)
		(typecase code
		  (symbol
		   (if (equal (symbol-name code)
			      (symbol-name type-var))
		       dtype
		       code))
		  (single-float (coerce code dtype))
		  (double-float (coerce code dtype))
		  (T
		   code)))
	    body))

(defun define-lisp-code-with (args
			      fname-with-prefix
			      dtype
			      body
			      declarations
			      type-specifier)
  `(defun ,fname-with-prefix (,@args)
     (locally
       ,@(unless (eql dtype 'double-float)
	(replace-lisp-code-with-dtype
	 declarations
	 type-specifier
	 dtype))
	
     ,@(replace-lisp-code-with-dtype
	body
	type-specifier
	dtype))))

(defun get-params (list)
  (reverse
   (delete-duplicates
    (flatten
     (loop for i fixnum upfrom 0 below (length list)
	   collect (let ((sym (nth i list)))
		     (typecase sym
		       (symbol
			(if (find sym `(&optional &rest &key &aux))
			    nil
			    sym))
		       (list
			(if (= (length sym) 2)
			    (car sym)
			    (get-params sym))))))))))

(defmacro define-with-typevar
    (function-name
     type-specifier
     (&rest args)
     &body body
     &aux (fnames (map
		   'list
		   #'(lambda (p)
		       (symb function-name p))
		   *dtype-prefixes*))
          (params (get-params args)))
  
  (multiple-value-bind (body declarations doc) (alexandria:parse-body `,body
								      :documentation t)
    `(progn
       ,@(loop for i fixnum upfrom 0 below (length fnames)
	       collect (define-lisp-code-with
			   params
			   (nth i fnames)
			 (nth i *dtype-cl-names*)
			 body
			 declarations
			 type-specifier))
       (defun ,function-name (,@args)
	 ,doc
	 (case mgl-mat:*DEFAULT-MAT-CTYPE*
	   (:half
	    (error "No implementation"))
	   (:float
	    (,(car fnames) ,@params))
	   (:double
	    (,(second fnames) ,@params))
	   (T
	    (error "No dtype")))))))

(defmacro dtypecase (&rest cases)
  `(case mgl-mat:*DEFAULT-MAT-CTYPE*
     ,@cases))
