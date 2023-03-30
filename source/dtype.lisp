
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
		  (T
		   code)))
	    body))


(defun keyword-parser (keywords)
  (let ((mode nil))
    
    ))

(defmacro define-with-typevar
    (function-name
     type-specifier
     (&rest args)
     &body body
     &aux (fnames (map
		   'list
		   #'(lambda (p)
		       (symb function-name p))
		   *dtype-prefixes*)))
  
  (multiple-value-bind (body declarations doc) (alexandria:parse-body `,body
								      :documentation t)
    (labels ((define-lisp-code-with (fname-with-prefix dtype)
	       `(,fname-with-prefix ()
				    ,@(replace-lisp-code-with-dtype
				      `(,@declarations ,@body)
				       type-specifier
				       dtype))))
      `(progn
	 (defun ,function-name (,@args)
	   ,doc
	   (labels (,@(loop for i fixnum upfrom 0 below (length fnames)
			    collect (define-lisp-code-with
					(nth i fnames)
					(nth i *dtype-cl-names*))))
	     (case mgl-mat:*DEFAULT-MAT-CTYPE*
	       (:half
		(error "No implementation"))
	       (:float
		(,(car fnames)))
	       (:double
		(,(second fnames)))
	       (T
		(error "No dtype")))))))))
