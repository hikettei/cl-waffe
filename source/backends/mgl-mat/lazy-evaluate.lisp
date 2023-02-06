
(in-package :cl-waffe.backends.mgl)


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
  `(return-from
    ,function-name
     (step-and-produce-lazy-eval
      ,tensor
      ,lisp-function
      ,args)))

(defun compile-and-run-lazy (tensor)
  (let ((tensor-data (data tensor)))
    (when (typep tensor-data 'function)
      (funcall tensor-data tensor nil t))))

(defun compile-and-step-lazy-evaluated-nodes
  (tensor-top
   lisp-function
   args)

  (let ((args-table (make-hash-table)))
    (print (generate-kernel-code args-table tensor-top lisp-function args))

    ))

(defun parse-argument (args-table tensor)
  (typecase tensor
    (function
     (multiple-value-bind
	   (last-tensor lisp-function args)
	 (funcall tensor nil nil nil nil t)
       (generate-kernel-code
	args-table
	last-tensor
	lisp-function
	args)))
    (waffetensor
     (parse-argument args-table (data tensor)))
    (T
     (typecase tensor
       (mat
	tensor)
       (T
	tensor)))))
	
      
(defun generate-kernel-code (args-table tensor lisp-function args)
  `(,lisp-function
    ,(parse-argument args-table tensor)
    ,@(map 'list #'(lambda (arg)
		     (parse-argument args-table arg))
	   args)))

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
  (let ((a (exp-test (!randn `(3 3)))))
    (const (add-test (exp-test a) a))))
