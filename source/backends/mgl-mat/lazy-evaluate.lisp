
(in-package :cl-waffe.backends.mgl)

; JIT Compiler

(defparameter *jit-compiled* (make-hash-table)
  "An hash table, jit-id -> jit-function-name")

(defparameter *fname-ids* (make-hash-table)
  "An hash table, function-name (i.e. exp etc...) -> id")

(defparameter *force-lazy-eval* nil
  "When t, every calculation in cl-waffe became lazy-eval. for debugging.")

(defun fname-get (symbol-name)
  "Translate symbol-name -> id, in order to reduce jit-id"
  (or (gethash symbol-name *fname-ids*)
      (prog1
	  (1+ (hash-table-count *fname-ids*))
	(setf (gethash symbol-name *fname-ids*)
	      (1+ (hash-table-count *fname-ids*))))))

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
  (declare (type waffetensor last-tensor)
	   (type symbol list-function)
	   (type list args))
  (labels ((LazyEvaluatedNodes (tensor-top return-shape? compile-and-step? &optional ignore? return-node-info)
	     (declare (ignore ignore?))
	     (cond
	       (return-shape?
		; How could i say last-tensor must be shape? it could be a number.
		(!shape last-tensor))
	       (return-node-info
		(values :lazy-eval last-tensor lisp-function args))
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
  "The kernel functions, like the shape is immutable, can be lazy evaluated.
tensor ... the first argument
args ... must be nil or cons. note that you must ignore the first argument

When the tensor isn't appropriate, do nothing."
  (declare (type list args))
  `(if (or (and cl-waffe.caches:*static-node-mode*
		(cl-waffe::waffetensor-thread-data ,tensor))
	   *force-lazy-eval*)
       ; Judge if the Tensor is in the Model's Iteration or in thread-data.
       (return-from
	,function-name
	 (step-and-produce-lazy-eval
	  ,tensor
	  ,lisp-function
	  (typecase ,args
	    (list ,args)
	    (T (error "return-lazy-eval: args must be list but got ~a" (type-of ,args))))))))

(defun compile-and-run-lazy (tensor)
  (declare (type waffetensor tensor))
  "If tensor is lazy evaluated, execute all nodes. otherwise return tensor."

  (print "==Node==")
  (fresh-line)
  (if (typep (data tensor) 'function)
      (display-all-nodes tensor))
  
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
  "generate kernel code based on tensor-top's backend.

Note jit-id: In Common Lisp, the maximum length of symbol is array-dimension-limit"
  (declare (ignore lisp-function args))
  (let* ((jit-id (make-string-output-stream)) ; jit-id is made for find compiled functions
	 (args-table (make-hash-table))
	 (result-code
	   (parse-argument
	    jit-id
	    args-table
	    tensor-top))
	 (jit-id (intern (get-output-stream-string jit-id) :keyword)))
    (if (use-cuda-p tensor-top)
	(error "Lazy eval doesn't support cuda environments") ; Todo: Support it
	(lisp-define-tmp-kernel
	 jit-id
	 args-table
	 result-code
	 tensor-top))))

(defun parse-argument (jit-id args-table tensor)
  "Parse args, if tensor=mat, register to args-table"
  (declare ;(optimize (speed 3))
	   (type stream jit-id))
  (typecase (data tensor)
    (function
     (multiple-value-bind
	   (node-type last-tensor lisp-function args)
	 (funcall (the function (data tensor)) tensor nil nil nil t)
       (declare (type list args))
       
       (if (eql node-type :lazy-eval)
	   (generate-kernel-code
	    jit-id
	    args-table
	    last-tensor
	    lisp-function
	    args)
	   (parse-argument
	    jit-id
	    args-table
	    (data tensor)))))
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
	(format jit-id "M")
	`(aref ,(cl-waffe::waffetensor-tensor-ident tensor)
	       (mod index ,(!size tensor))))
       (T
	(format jit-id "O")
	(data tensor))))))
      
(defun generate-kernel-code (jit-id args-table tensor lisp-function args)
  "The top level of generating code.
jit-id is a stream"
  (declare (optimize (speed 3))
	   (type stream jit-id))
  (format jit-id ".~a" (fname-get lisp-function)) ;replace . with ( and , with )
  (prog1
      `(,lisp-function
	,(parse-argument jit-id args-table tensor)
	,@(map 'list #'(lambda (arg)
			 (parse-argument jit-id args-table arg))
	       args))
    (format jit-id ",")))

(defun lisp-execute-tmp-kernel
    (args-table
     any-tensor
     &key (jit-function-id nil))
  ;(declare (optimize (speed 3)))
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
	   ;(print "===New Code Loaded===")
	   ;(print jit-function-id)
	   ;(dolist (m mat-inputs)
	   ;  (print (const m)))
	   (apply-jit
	    jit-function-id
	    `(,(mat-size out) ,out ,@mat-inputs))
	   ;(print "RESULT")
	   ;(print (const out))
	   out)
	  (T (error "cl-waffe.backends.mgl:JIT -> couldn't find jit-id")))))))

(defun lisp-define-tmp-kernel (jit-id
			       args-table
			       code
			       any-tensor
			       &aux (jit-ident (gensym "JitFunction")))
  (declare (optimize (speed 3)))
  "do define-lisp-kernel and execute it.
Return: compiled-function's id, out"
  (if (gethash jit-id *jit-compiled*)
      (return-from lisp-define-tmp-kernel
	  (lisp-execute-tmp-kernel args-table
				   any-tensor
				   :jit-function-id
				   (gethash jit-id *jit-compiled*))))
  
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

	  ;let ((out (make-mat (!shape any-tensor))))
	  ;; Todo: SetfAref -> マクロにする、計算ノード保持するように。
	  ;; Todo: any-tensorが不要ならany-tensorに書き込む
          ;(print "====New Code Compiled==")
	  ;(print jit-id)
	  ;(print kernel-code)

	  (eval kernel-code)
	  ;(dolist (v symbols)
	  ;  (print v))
	  ;(print (const out))
	  ;(dolist (v mat-inputs)
	  ;  (print (data (!aref (const v) 0))))
	  (setf (gethash jit-id *jit-compiled*)
		jit-ident)
	  (apply-jit
	   jit-ident
	   `(,(mat-size out) ,out ,@mat-inputs))
	 ; (print "Result")
	  ;(print (const out))
	  out)))))

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
