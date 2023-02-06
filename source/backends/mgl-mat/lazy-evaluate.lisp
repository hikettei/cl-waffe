
(in-package :cl-waffe.backends.mgl)

; JIT Compiler

(defun display-all-nodes (tensor &optional (indent 0))
  (let ((variables (cl-waffe::waffetensor-variables tensor))
	(state     (cl-waffe::waffetensor-state tensor)))
    (dotimes (i indent)
      (format t " "))
    (format t "[Tensor: ~a]~%" state)

    (dolist (v variables)
      (if (not (null v))
	  (display-all-nodes v (+ indent 2))))))

(defun step-and-produce-lazy-eval (last-tensor lisp-function args)
  "Lazy eval's format is following: (free-args shape? return-calculated-value?)"
  (labels ((LazyEvaluatedNodes (tensor-top return-shape? compile-and-step? &optional ignore? return-node-info)
	     (declare (ignore ignore?))
	     (cond
	       (return-shape?
		nil)
	       (return-node-info
		(values last-tensor lisp-function args))
	       (compile-and-step?
		(compile-and-step-lazy-evaluated-nodes
		 tensor-top
		 lisp-function
		 args)))))
    #'LazyEvaluatedNodes))
  
(defmacro return-and-lazy-eval
    (function-name
     lisp-function
     tensor
     args)
  "The kernel functions, like the shape is immutable, can be lazy evaluated."
  `(return-from
    ,function-name
     (step-and-produce-lazy-eval
      ,tensor
      ,lisp-function
      ,args)))

(defun compile-and-run-lazy (tensor)
  (when (typep (data tensor) 'function)
    (funcall (data tensor) tensor nil t)))

(defun compile-and-step-lazy-evaluated-nodes
  (tensor-top
   lisp-function
   args)

  (let* ((args-table (make-hash-table))
	 (result-code
	   (generate-kernel-code args-table tensor-top lisp-function args)))
    (if (use-cuda-p tensor-top)
	(error "Lazy eval doesn't support cuda environments")
	(lisp-define-tmp-kernel args-table result-code))))

(defun parse-argument (args-table tensor)
  (typecase (data tensor)
    (function
     (multiple-value-bind
	   (last-tensor lisp-function args)
	 (funcall (data tensor) tensor nil nil nil t)
       (generate-kernel-code
	args-table
	last-tensor
	lisp-function
	args)))
    (T
     (typecase (data tensor)
       (mat
	(if (null (cl-waffe::waffetensor-tensor-ident tensor))
	    (setf (cl-waffe::waffetensor-tensor-ident tensor)
		  (gensym "KernelArgs")))
	(setf (gethash
	       (cl-waffe::waffetensor-tensor-ident tensor)
	       args-table)
	      (data tensor))
	`(aref ,(cl-waffe::waffetensor-tensor-ident tensor) index))
       (T
	tensor)))))
      
(defun generate-kernel-code (args-table tensor lisp-function args)
  `(,lisp-function
    ,(parse-argument args-table tensor)
    ,@(map 'list #'(lambda (arg)
		     (parse-argument args-table arg))
	   args)))

(defun lisp-define-tmp-kernel (args-table code)
  (macrolet ((def-dynamic-kernel (args body)
	       `(progn
		  `(mgl-mat:define-lisp-kernel (.tmp-kernel)
		    ,,args
		     (loop for index fixnum upfrom 0 below size
			   do (setf (aref out index) ,,body))))))
    (let ((symbols nil))
      (maphash #'(lambda (key val)
		   (declare (ignore val))
		   (push `(,key :mat :input) symbols))
	       args-table)
      (setq symbols `((size fixnum)
		      (out :mat :output)
		      ,@(reverse symbols)))
      (def-dynamic-kernel symbols code))))

(defun add-test (tensor x)
  (return-and-lazy-eval add-test
			'+
			tensor
			`(,x)))

(defun exp-test (tensor)
  (return-and-lazy-eval exp-test
			'exp
			tensor
			nil))

(defun return-test-node ()
  (let ((a (const (exp-test (!randn `(3 3))))))
    (const (add-test
	    (const (exp-test a))
	    a))))
