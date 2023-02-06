
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
		(!shape last-tensor))
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

(defun compile-and-run-lazy (tensor &key (jit-id nil))
  "If tensor is lazy evaluated, execute all nodes. otherwise return tensor."
  (setf (cl-waffe::waffetensor-jit-id tensor) jit-id)
  (if (typep (data tensor) 'function)
      (funcall
       (data tensor)
       tensor
       nil
       t)
      tensor))

(defun compile-and-step-lazy-evaluated-nodes
  (tensor-top
   lisp-function
   args)
  "generate kernel code based on tensor-top's backend."
  (let* ((args-table (make-hash-table))
	 (result-code
	   (generate-kernel-code args-table tensor-top lisp-function args)))
    (if (use-cuda-p tensor-top)
	(error "Lazy eval doesn't support cuda environments")
	(lisp-define-tmp-kernel
	 args-table
	 result-code
	 tensor-top))))

(defun parse-argument (args-table tensor)
  "Parse args, if tensor=mat, register to args-table"
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
  "The top level of generating code."
  `(,lisp-function
    ,(parse-argument args-table tensor)
    ,@(map 'list #'(lambda (arg)
		     (parse-argument args-table arg))
	   args)))

(defun lisp-execute-tmp-kernel
  (args-table
   any-tensor
   &key (jit-function-id nil))
  (print "A")
  (macrolet ((apply-jit (jit-id args)
	       `(apply (intern (symbol-name ,jit-id)) ,args)))
    (let ((mat-inputs nil))
      (maphash #'(lambda (key val)
		   (declare (ignore key))
		   (push val mat-inputs))
	       args-table)

      (setq mat-inputs (reverse mat-inputs))
      
      (cl-waffe.caches:with-cache (out any-tensor)
	(if (cl-waffe::waffetensor-thread-data any-tensor)
	    (incf (cl-waffe::waffenodethread-cache-n
		   (cl-waffe::waffetensor-thread-data any-tensor))
		  1))
	(cond
	  (jit-function-id
	   (apply-jit
	    jit-function-id
	    `(,(mat-size out) ,out ,@mat-inputs))
	   (values jit-function-id out)))))))

(defun lisp-define-tmp-kernel (args-table
			       code
			       any-tensor
			       &aux (jit-ident (gensym "JitFunction")))
  ;(declare (optimize))
  "do define-lisp-kernel and execute it.
Return: compiled-function's id, out"
  (if (not (null (cl-waffe::waffetensor-jit-id any-tensor)))
      (multiple-value-bind (id out)
	  (lisp-execute-tmp-kernel args-table
				   any-tensor
				   :jit-function-id
				   (cl-waffe::waffetensor-jit-id any-tensor))
	(return-from lisp-define-tmp-kernel (values id out))))
  
  (macrolet ((def-dynamic-kernel (args body)
	       `(progn
		  `(mgl-mat:define-lisp-kernel (,(intern (symbol-name jit-ident)))
		    ,,args
		     (loop for index fixnum upfrom 0 below size
			   do (setf (aref out index) ,,body)))))
	     (apply-jit (jit-id args)
	       `(apply (intern (symbol-name ,jit-id)) ,args)))
    (let ((symbols nil)
	  (mat-inputs nil))
      (maphash #'(lambda (key val)
		   (push `(,key :mat :input) symbols)
		   (push val mat-inputs))
	       args-table)
      
      (setq symbols `((size fixnum)
		      (out :mat :output)
		      ,@(reverse symbols)))

      (setq mat-inputs (reverse mat-inputs))
      (let* ((kernel-code (def-dynamic-kernel symbols code)))
	(cl-waffe.caches:with-cache (out any-tensor)
	  (if (cl-waffe::waffetensor-thread-data any-tensor)
	      (incf (cl-waffe::waffenodethread-cache-n
		     (cl-waffe::waffetensor-thread-data any-tensor))
		    1))
	  (eval kernel-code)
	  (apply-jit
	   jit-ident
	   `(,(mat-size out) ,out ,@mat-inputs))
	  (values jit-ident out))))))

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

(defun return-test-node (tensor)
  (let ((a (const (exp-test tensor))))
    (const (add-test
	    (const (exp-test a))
	    a))))

(defun run-orig (a)
  (time (dotimes (i 1000)
	     (progn
	       (!add (!exp (!exp a)) (!exp a))
	       nil))))
