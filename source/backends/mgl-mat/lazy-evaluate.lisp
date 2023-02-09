
(in-package :cl-waffe.backends.mgl)

; JIT Compiler

(defparameter *jit-compiled* (make-hash-table)
  "An hash table, jit-id -> jit-function-name")

(defparameter *fname-ids* (make-hash-table)
  "An hash table, function-name (i.e. exp etc...) -> id")

(defparameter *force-lazy-eval* nil
  "When t, every calculation in cl-waffe became lazy-eval. for debugging.")

(defparameter *verbose* nil
  "When t, jit compiler and cl-waffe.caches can output logs. for debugging.")

(defparameter *ignore-jit-max-len* 3
  "When computation node is built and that length <= *ignore-jit-max-len* call backpoint and call blas api.")

; utils
(defun mkstr (&rest args)
  "concatenates args by printing into string"
  (with-output-to-string (s)
    (dolist (a args) (princ a s))))

(defun symb (&rest args)
  "interns the mkstr output/returns as symbol"
  (values (intern (apply #'mkstr args))))

(defmacro mat-size-symbol (sym)
  `(symb 'n ,sym))

(defun fname-get (symbol-name)
  "Translate symbol-name -> id, in order to reduce jit-id"
  (or (gethash symbol-name *fname-ids*)
      (prog1
	  (1+ (hash-table-count *fname-ids*))
	(setf (gethash symbol-name *fname-ids*)
	      (1+ (hash-table-count *fname-ids*))))))

(defun get-function-type (func)
  (funcall func nil nil nil nil t))

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
		(let* ((first-shape (!shape last-tensor))
		       (args-shape (map 'list
					#'(lambda (x)
					    (typecase (data x)
					      (function nil)
					      (T (!shape x))))
					args))
		       (max-size (apply #'max
					(apply #'* first-shape)
					(map 'list
						   #'(lambda (x) (apply #'* x))
						   args-shape)))
		       (max-pos (position
				 max-size
				 `(,first-shape ,@args-shape)
				 :test #'(lambda (size x) (= size (apply #'* x))))))
		  (nth max-pos `(,first-shape ,@args-shape))))
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
  "If tensor is lazy evaluated, execute all nodes. otherwise return tensor."
  (declare (type waffetensor tensor))
  ;(when *verbose*
  ;  (format t "~%JIT Found Compileable node:~%")
  ;  (if (typep (data tensor) 'function)
;	(display-all-nodes tensor)))
  
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
  (declare (optimize (speed 3))
	   (ignore lisp-function args))
  (let* ((jit-id (make-string-output-stream)) ; jit-id is made for find compiled functions
	 (args-table (make-hash-table))
	 (mat-size-table (make-hash-table))
	 (result-code
	   (parse-argument
	    jit-id
	    args-table
	    mat-size-table
	    tensor-top))
	 (jit-id (intern (get-output-stream-string jit-id) :keyword)))
    (if (use-cuda-p tensor-top)
	(error "Lazy eval doesn't support cuda environments") ; Todo: Support it
	(lisp-define-tmp-kernel
	 jit-id
	 args-table
	 mat-size-table
	 result-code
	 tensor-top))))

(defun parse-argument (jit-id args-table mat-size-table tensor)
  "Parse args, if tensor=mat, register to args-table"
  (declare (optimize (speed 3))
	   (type stream jit-id))
  (typecase (data tensor)
    (function
     ; when tensor has unevaluated nodes?
     (multiple-value-bind
	   (node-type last-tensor lisp-function args)
	 (funcall (the function (data tensor)) tensor nil nil nil t)
       (declare (type list args))
       (if (eql node-type :lazy-eval)
	   ; the function is lazy-eval, explore them.
	   (generate-kernel-code
	    jit-id
	    args-table
	    mat-size-table
	    last-tensor
	    lisp-function
	    args)
	   ; otherwise, regard the function as mat (cached or transposed).
	   (parse-argument
	    jit-id
	    args-table
	    mat-size-table
	    (data tensor)))))
    (T
     (typecase (data tensor)
       (mat
	; when tensor is the end of node?
	(if (null (cl-waffe::waffetensor-tensor-ident tensor))
	    (setf (cl-waffe::waffetensor-tensor-ident tensor) (gensym "K")))
	(setf (gethash
	       (cl-waffe::waffetensor-tensor-ident tensor)
	       args-table)
	      (data tensor))
	(setf (gethash
	       (mat-size-symbol
		(cl-waffe::waffetensor-tensor-ident tensor))
	       mat-size-table)
	      (mat-size (data tensor))) ; (data tensor) is supposed to be mat. (the end of node.)
	(format jit-id "M")
	`(aref ,(cl-waffe::waffetensor-tensor-ident tensor)
	       (mod index ,(mat-size-symbol (cl-waffe::waffetensor-tensor-ident tensor)))))
       (T
	(if (null (cl-waffe::waffetensor-tensor-ident tensor))
	    (setf (cl-waffe::waffetensor-tensor-ident tensor) (gensym "K")))
	(setf (gethash
	       (cl-waffe::waffetensor-tensor-ident tensor)
	       args-table)
	      (data tensor))
	(format jit-id "O")
	(cl-waffe::waffetensor-tensor-ident tensor))))))
      
(defun generate-kernel-code (jit-id args-table mat-size-table tensor lisp-function args)
  "The top level of generating code.
jit-id is a stream"
  (declare (optimize (speed 3))
           (type stream jit-id))
  ; in jit-id, (f 1 2) -> .fOO, to reduce the length of chars
  (format jit-id ".~a" (fname-get lisp-function))
  (prog1
      `(,lisp-function
	,(parse-argument jit-id args-table mat-size-table tensor)
	,@(map 'list #'(lambda (arg)
			 (parse-argument jit-id args-table mat-size-table arg))
	       args))
    (format jit-id ",")))

(defun lisp-execute-tmp-kernel
    (args-table
     mat-size-table
     any-tensor
     &key (jit-function-id nil))
  (declare (optimize (speed 3) (space 0)))
  (macrolet ((apply-jit (jit-id args)
	       `(apply (intern (symbol-name ,jit-id)) ,args)))
    (let ((mat-inputs nil)
	  (mat-ninputs nil))
      (maphash #'(lambda (key val)
		   (declare (ignore key))
		   (push val mat-inputs))
	       args-table)

      (maphash #'(lambda (key val)
		   (declare (ignore key))
		   (push val mat-ninputs))
	       mat-size-table)

      (setq mat-inputs `(,@(reverse mat-inputs)
			 ,@(reverse mat-ninputs)))
      ;(warranty any-tensor)
      (let ((out (make-mat (!shape any-tensor))));cl-waffe.caches:with-cache (out any-tensor)
	(if (cl-waffe::waffetensor-thread-data any-tensor)
	    (incf (cl-waffe::waffenodethread-cache-n
		   (cl-waffe::waffetensor-thread-data any-tensor))
		  1))
	(cond
	  (jit-function-id
	   
	   ;(when *verbose*
	   ;  (format t "~%JIT Loaded Compiled Function: ~a~%" jit-function-id))	   
	   (apply-jit
	    jit-function-id
	    `(,(mat-size out) ,out ,@mat-inputs))
	   out)
	  (T ; when jit-function-id is not found? (in that case hash-table could be modified)
	   (error "cl-waffe.backends.mgl:isp-execute-tmp-kernel (JIT) -> couldn't find jit-id")))))))

(defun lisp-define-tmp-kernel (jit-id
			       args-table
			       mat-size-table
			       code
			       any-tensor
			       &aux (jit-ident (gensym "JitFunction")))
  ;(declare (optimize (speed 3)))
  "do define-lisp-kernel and execute it.
Return: compiled-function's id, out"
  (declare (optimize (speed 3) (space 0))
	   (type list code))
  (if (gethash jit-id *jit-compiled*)
      (return-from lisp-define-tmp-kernel
	(lisp-execute-tmp-kernel args-table
				 mat-size-table
				 any-tensor
				 :jit-function-id
				 (gethash jit-id *jit-compiled*))))
  
  (macrolet ((def-dynamic-kernel (args body)
	       `(if (and (<= (length code) (the fixnum *ignore-jit-max-len*))
			 (<= (length ,args) (+ 1 1 (* 2 2))))
		    ; ignore jit like: (+ a b) or (exp a)
		   (progn
		     `(defun ,(intern (symbol-name jit-ident))
			,(map 'list #'car ,args)
			(declare (ignore
				  size
				  ,@(map
				     'list
				     #'car
				     (remove-if
				       (lambda (x)
					 (eql (second x) :mat))
				       ,args))))
			,(let* ((mat-args (remove-if-not
					   (lambda (x)
					     (eql (second x) :mat))
					   ,args))
				(mat-args (map 'list #'car mat-args)))
			   `(progn
			      (axpy! 1.0 ,(car mat-args) ,(second mat-args))
			      ,(third mat-args)))))
		   (progn
		      `(mgl-mat:define-lisp-kernel
			   (,(intern (symbol-name jit-ident)))
			   ,,args
			 (loop for index fixnum upfrom 0 below size
			       do (setf (aref out index) ,,body))))))
	     (apply-jit (jit-id args)
	       `(apply (intern (symbol-name ,jit-id)) ,args)))
    ; collecting inputs
    (let ((symbols nil)  ; args for mat, obj
	  (nsymbols nil) ; args for mat-size
	  (mat-inputs nil) ; the list of values
	  (mat-ninputs nil)) ; the list of value's size
      
      (maphash #'(lambda (key val)
		   (push `(,key ,@(typecase val
				   (mat
				    `(:mat :input))
				   (T
				    `(,(type-of val)))))
			 symbols)
		   (push val mat-inputs))
	       args-table)


      (maphash #'(lambda (key val)
		   (push `(,key fixnum) nsymbols)
		   (push val mat-ninputs))
	       mat-size-table)
      
      (setq symbols `((size fixnum)
		      (out :mat :output)
		      ,@(reverse symbols)
		      ,@(reverse nsymbols)))

      (setq mat-inputs `(,@(reverse mat-inputs)
			 ,@(reverse mat-ninputs)))
      (let* ((kernel-code (def-dynamic-kernel symbols code)))
	;(warranty any-tensor)
	(let ((out (make-mat (!shape any-tensor))));cl-waffe.caches:with-cache (out any-tensor)
	  (if (cl-waffe::waffetensor-thread-data any-tensor)
	      (incf (cl-waffe::waffenodethread-cache-n
		     (cl-waffe::waffetensor-thread-data any-tensor))
		    1))

	  ;; Todo: SetfAref -> マクロにする、計算ノード保持するように。
	  ;; Todo: any-tensorが不要ならany-tensorに書き込む

	  (when *verbose*
	    (format t "~%JIT Compiled New function ~a~%" jit-ident)
	    (print kernel-code)
	    (fresh-line))

	  ; eval define-lisp-kernel/define-cuda-kernel
	  (eval kernel-code)

	  ; make compiled function recallable.
	  (setf (gethash jit-id *jit-compiled*) jit-ident)

	  ; execute it and write result to out.
	  (apply-jit
	   jit-ident
	   `(,(mat-size out) ,out ,@mat-inputs))
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
