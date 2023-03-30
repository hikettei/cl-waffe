
(in-package :cl-waffe.backends.mgl)

#|
Here's
  1. Broadcasting
  2. Operations using mgl-mat
|#

(defmacro will-be-destructed (tensor)
  `(waffetensor-thread-data ,tensor))

(defmacro wanted-to-be-destructed? (tensor)
  (cl-waffe::waffetensor-is-next-destruct? tensor))

(defun create-thread-idx (thread-info &optional (ident ""))
  "Thread format: <Thread_IDx>+<Count_N>"
  (if thread-info
      (intern (format nil "~a+~a~a"
		      ident
		      (cl-waffe::waffenodethread-thread-idx thread-info)
		      (cl-waffe::waffenodethread-cache-n thread-info))
	      :keyword)
      (gensym)))

(defun decide-out-buffer (out args enable-optim copy?)
  (declare (optimize (speed 3) (safety 0))
	   (inline decide-out-buffer2
		   decide-out-buffer3))
  (let ((args (if (typep args 'waffetensor)
		  args
		  (sysconst args))))
    (value args)
    (if (null out)
	(decide-out-buffer3 out (data args) enable-optim copy?)
	(typecase args
	  (waffetensor
	   (decide-out-buffer2 out (data args) enable-optim copy?))))))
     
(defun decide-out-buffer2 (out
			   args
			   enable-optim
			   copy?)
  (declare (optimize (speed 3) (safety 0)))
  (if (not (null (waffetensor-thread-data out)))
      (let* ((thread-info (waffetensor-thread-data out))
	     (idx (create-thread-idx thread-info)))
	(value out)
	(with-cache (result out :place idx :copy copy?)
	  result))
      (decide-out-buffer nil args enable-optim copy?)))

(defun decide-out-buffer3 (out
			   args
			   enable-optim
			   copy?)
  (declare (optimize (speed 3) (safety 0))
	   (ignore out))
  (if enable-optim
      args
      (if copy?
	  (copy-mat args)
	  (make-mat (mat-dimensions args)))))

(declaim (ftype (function (mgl-mat:mat fixnum &key (:axis fixnum)) mgl-mat:mat) mgl-repeat))
(defun mgl-repeat (tensor n &key axis)
  (declare (optimize (speed 3))
	   (type mgl-mat:mat tensor)
	   (type fixnum n axis))
  (let* ((axis (if (>= axis 0)
 		   axis
		   (+ (length (the list (mat-dimensions tensor))) axis)))
	 (new-tensor-dim (loop for i fixnum upfrom 0 below (length (the list (mat-dimensions tensor)))
			       if (= i axis)
				 collect (the fixnum (* n (the fixnum (mat-dimension tensor i))))
			       else		
				 collect (mat-dimension tensor i)))
	 (new-tensor (!zeros new-tensor-dim))
	 (base-tensor (sysconst tensor)))
    (data (cl-waffe::%saref
	   new-tensor
	   base-tensor
	   t))))

;(declaim (ftype (function (mgl-mat:mat waffesupporteddatatype) mgl-mat:mat) trasposedmgl-full-like mgl-full-like))
(defun mgl-full-like (tensor value)
  (declare (optimize (speed 3) (safety 0) (debug 0))
	   (type mat tensor))
  (format t "Warning: making new tensor~%")
  (make-mat (mat-dimensions tensor)
	    :initial-element value))

(defun transposed-mgl-full-like (tensor value)
  (declare (optimize (speed 3) (safety 0) (debug 0))
	   (type mat tensor))
  (format t "Warning: making new tensor~%")
  (let ((dims (mat-dimensions tensor)))
    (declare (type cons dims))
    (make-mat (reverse dims) :initial-element value)))


(defparameter *v2v-operations* `(:add :sub :mul :div :dot :matmul))
(defparameter *abort-delay-instruction* :matmul)


(defmacro define-onearg-fn (name function)
  `(define-with-typevar ,name u (mat)
     (with-facet (r (mat 'backing-array :direction :input))
       (declare (type (simple-array u) r))
       (loop for i fixnum upfrom 0 below (mat-size mat)
	     do (setf (aref r i) (,function (aref r i))))
       r)))

(defun lazy-eval-transpose (tensor args)
  "Lazy eval's format is following: (free-args shape? return-calculated-value?)"
  (declare (ignore args))
  (if (typep tensor 'waffetensor)
      (error "KernelError: lazy-eval-transpose -> tensor must not be waffetensor."))
  (labels ((LazyTranspose (given-tensor
			   return-shape?
			   compile-and-step?
			   &optional
			     ignore?
			     return-node-info)
	     (declare (ignore given-tensor))
	     #+sbcl(declare (sb-ext:muffle-conditions cl:warning)) ; how to disable warnings T_T
					; given-tensor is always lazytranspsoed.
	     (cond
	       (ignore?
		nil)
	       (return-shape?
					; Return transposed dims (for 2d only) for 3d is todo.
		(let ((result (!shape (sysconst tensor))))
		  (declare (type list result))
		  (case (length result)
		    (1 (reverse result))
		    (2 (reverse result))
		    (T `(,@(subseq result 0 (- (length result) 2))
			 ,@(reverse (subseq result (- (length result) 2)
					    (length result))))))))
	       (return-node-info
		(values :lazy-transpose nil nil nil))
	       (compile-and-step?
		(let ((tns (sysconst (compile-and-run-lazy (sysconst tensor)))))
		  (if (>= (!dims tns) 3)
		      (progn
			(format t "Warning: Transpose1 is called with 3d Tensor which is super slow...(To Fix)~%")
			(data (!transpose1 tns))) ; This is bottleneck and need to be fixed.
		      (transpose (data tns)))))
	       (T
		 ; The Last Transpose is skipped, returning untransposed tensor
		 ; this block will be called by (value ~ :ignore-transpose t)
		 ; so, if tensor should function, this will be evaluated.
		(value (sysconst tensor))))))
    #'LazyTranspose))

(defmacro deliv-delay (tensor func &rest args)
  (declare (ignore func))
  (lazy-eval-transpose tensor args))

(defun next-delay (delay state)
  (if (typep delay 'function)
      (funcall delay nil nil state)
      delay))

(defmacro abort-delay (delay)
  `(next-delay ,delay nil))

(defmacro receive-delay (delay)
  `(next-delay ,delay t))

(defun ensure-shape (ope x)
  (declare (type keyword ope))
  (if (find ope *v2v-operations*)
      (if (or (typep x 'mat) (typep x 'function))
	  (if (eq ope *abort-delay-instruction*)
	      (abort-delay x)
	      (receive-delay x))
	  (if (eq ope :matmul)
	      (transposed-mgl-full-like x 0)
	      (mgl-full-like x 0)))))

(defun infomation ())

(declaim (ftype (function (boolean waffetensor waffetensor) mgl-mat:mat)
		inv-tensor
		sqrt-tensor
		log-tensor
		tanh-tensor
		exp-tensor))

					; kernel function must be following:
					; fname (enable-optimize? args-tensor1 args-tensor2 ... mat1 mat2 ....)
					; Otherwise ignore jit can't return correctly.
					; Todo: Write macro in order to define this.

(defmacro define-waffe-kernel (name
			       args
			       args-mat
			       &key
				 (ignore-optimize nil)
				 (jit nil)
				 (mat-mat nil)
				 (scal-mat nil)
				 (mat-scal nil))
  "(define-waffe-kernel-function add-tensor (a b) (x y)
      :jit '+)
      :mat-mat ~
      :scal-mat ~
      :mat-scal ~"
  `(defun ,name (enable-optimize? ,@args &key (output nil) (overwrite nil))
     ,(unless ignore-optimize
	`(declare (optimize (speed 3))
		  (type boolean enable-optimize? overwrite)
		  (type waffetensor ,@args)))
     ,(unless (null jit)
	 ; place jit trigger.
	`(return-and-lazy-eval
	  ,name
	  ',jit
	  ,(car args)
	  (list ,@(cdr args))))
     
     ; if jit triggered, the form below never called.

     (macrolet ((get-out-buffer (tensor &key (copy nil))
		  `(cond
		     (overwrite (value ,tensor))
		     (output
		      (if ,copy
			  (copy! (value ,tensor) output))
		      output)
		     ((cl-waffe::waffetensor-is-next-destruct? ,tensor)
		      (value ,tensor))
		     (T (decide-out-buffer ,tensor (value ,tensor) enable-optimize? ,copy)))))
       
       (let (,@(map 'list (lambda (target val)
			    `(,target (value ,val)))
		    args-mat args))
	 (cond
	   ((or (= (length ',args) 1)
		(and (= (length ',args) 2)
		     (and (typep ,(car args-mat) 'mat)
			  (typep ,(second args-mat) 'mat)))
		(and (>= (length ',args) 3)))
	    ,@mat-mat)
	   ((or (= (length ',args) 1)
		(and (= (length ',args) 2)
		     (and (not (typep ,(car args-mat) 'mat))
			  (typep ,(second args-mat) 'mat)))
		(and (>= (length ',args) 3)))
	    ,(if (null scal-mat)
		 `(progn ,@mat-mat)
		 `(progn ,@scal-mat)))
	   ((or (= (length ',args) 1)
		(and (= (length ',args) 2)
		     (and (typep ,(car args-mat) 'mat)
			  (not (typep ,(second args-mat) 'mat))))
		(and (>= (length ',args) 3)))
	    ,(if (null mat-scal)
		 `(progn ,@mat-mat)
		 `(progn ,@mat-scal)))
	   (T (error "define-waffe-kernel: arguments didn't hit.")))))))

(define-waffe-kernel kernel-add (x y) (x1 y1)
  :jit +
  :mat-scal ((let ((o (get-out-buffer x :copy t)))
	       (.+! y1 o)))
  :scal-mat ((let ((o (get-out-buffer y :copy t)))
	       (.+! x1 o)))
  :mat-mat ((if (equal (!shape x) (!shape y))
		(cond
		  ((will-be-destructed x)
		   (let ((o (get-out-buffer x :copy t)))
		     (axpy! 1.0 y1 o)))
		  ;((will-be-destructed y)
		   ;(axpy! 1.0 x1 (get-out-buffer y :copy t)))
		  (T (axpy! 1.0 y1 (get-out-buffer x :copy t))))
		(broadcasting-apply :+ x y))))

(define-waffe-kernel kernel-sub (x y) (x1 y1)
  :jit -
  :mat-scal ((let ((o (get-out-buffer x :copy t)))
	       (.+! (the single-float (* -1.0 (the single-float y1)))
		    o)))
  :scal-mat ((let ((o (get-out-buffer y :copy t)))
	       (.+! x1
		    (scal! -1.0 o))))
  :mat-mat ((if (equal (!shape x) (!shape y))
		(cond
		  ((will-be-destructed x)
		   (let ((o (get-out-buffer x :copy t)))
		     (axpy! -1.0 y1 o)))
		  ;((will-be-destructed y)
		  ; (axpy! 1.0 x1 (scal! -1.0 (get-out-buffer y :copy t))))
		  (T (axpy! -1.0 y1 (get-out-buffer x :copy t))))
		(broadcasting-apply :- x y))))

(define-waffe-kernel kernel-mul (x y) (x1 y1)
  :jit *
  :mat-scal ((let ((o (get-out-buffer x :copy t)))
	       (scal! y1 o)))
  :scal-mat ((let ((o (get-out-buffer y :copy t)))
	       (scal! x1 o)))
  :mat-mat ((if (equal (!shape x) (!shape y))
		(cond
		  ((will-be-destructed x)
		   (let ((o (get-out-buffer x :copy nil)))
		     (geem! 1.0 x1 y1 0.0 o)))
		  ;((will-be-destructed y)
		  ; (geem! 1.0 x1 y1 0.0 (get-out-buffer y :copy nil)))
		  (T (geem! 1.0 x1 y1 0.0 (get-out-buffer x :copy nil))))
		(broadcasting-apply :* x y))))

(define-waffe-kernel kernel-inv (x) (x1)
  :jit /
  :mat-mat ((.inv! (get-out-buffer x :copy t))))

(define-waffe-kernel kernel-div (x y) (x1 y1)
  :jit /
  :mat-scal ((let ((o (get-out-buffer x :copy t)))
	       (scal! (/ y1) o)))
  :scal-mat ((unless (= x1 1)
	       (error "cl-waffe.backends.mgl:kernel-inv excepts x1 to be 1"))
	     (let ((o (get-out-buffer y :copy t)))
	       (.inv! o))))

(defun dot-tensor (enable-optimize? out x y)
  (declare (ignore enable-optimize? out))
  (mgl-mat:dot (value x) (value y)))

(defun is-transpose? (tensor)
  (declare (type waffetensor tensor))
  (typecase (data tensor)
    (function
     (multiple-value-bind
	   (node-type)
	 (funcall (data tensor) nil nil nil nil t)
       (eql node-type :lazy-transpose)))
    (T nil)))


(define-waffe-kernel kernel-log (x) (x1)
  :jit log
  :mat-mat ((.log! (get-out-buffer x :copy t))))

(define-waffe-kernel kernel-exp (x) (x1)
  :jit exp
  :mat-mat ((.exp! (get-out-buffer x :copy t))))

(define-waffe-kernel kernel-sqrt (x) (x1)
  :jit sqrt
  :mat-mat ((.sqrt! (get-out-buffer x :copy t))))

(define-waffe-kernel kernel-pow (x n) (x1 n1)
  :jit pow
  :mat-mat ((.expt! (get-out-buffer x :copy t) n1)))

(define-waffe-kernel kernel-sin (x) (x1)
  :jit sin
  :mat-mat ((.sin! (get-out-buffer x :copy t))))

(define-waffe-kernel kernel-cos (x) (x1)
  :jit cos
  :mat-mat ((.cos! (get-out-buffer x :copy t))))

(define-waffe-kernel kernel-tan (x) (x1)
  :jit tan
  :mat-mat ((.tan! (get-out-buffer x :copy t))))


(define-waffe-kernel kernel-sinh (x) (x1)
  :jit sinh
  :mat-mat ((.sinh! (get-out-buffer x :copy t))))

(define-waffe-kernel kernel-cosh (x) (x1)
  :jit cosh
  :mat-mat ((.cosh! (get-out-buffer x :copy t))))

(define-waffe-kernel kernel-tanh (x) (x1)
  :jit tanh
  :mat-mat ((.tanh! (get-out-buffer x :copy t))))


(define-onearg-fn apply-asin asin)
(define-onearg-fn apply-acos acos)
(define-onearg-fn apply-atan atan)

(define-waffe-kernel kernel-asin (x) (x1)
  :ignore-optimize t
  :jit asin
  :mat-mat ((apply-asin (get-out-buffer x :copy t))))

(define-waffe-kernel kernel-acos (x) (x1)
  :ignore-optimize t
  :jit acos
  :mat-mat ((apply-acos (get-out-buffer x :copy t))))

(define-waffe-kernel kernel-atan (x) (x1)
  :ignore-optimize t
  :jit atan
  :mat-mat ((apply-atan (get-out-buffer x :copy t))))

(define-onearg-fn apply-asinh asinh)
(define-onearg-fn apply-acosh acosh)
(define-onearg-fn apply-atanh atanh)

(define-waffe-kernel kernel-asinh (x) (x1)
  :ignore-optimize t
  :jit asinh
  :mat-mat ((apply-asinh (get-out-buffer x :copy t))))

(define-waffe-kernel kernel-acosh (x) (x1)
  :ignore-optimize t
  :jit acosh
  :mat-mat ((apply-acosh (get-out-buffer x :copy t))))

(define-waffe-kernel kernel-atanh (x) (x1)
  :ignore-optimize t
  :jit atanh
  :mat-mat ((apply-atanh (get-out-buffer x :copy t))))

(defun compare-tensor (enable-optim out x y)
  (declare (optimize (speed 3) (safety 0))
	   (type boolean enable-optim)
           (type waffetensor out x y))
  
  (let ((o (decide-out-buffer out x enable-optim t)))
    (mgl-mat:.<! (value y) o)))

(defun sum-tensor (is-first-time-call? out x y)
  (declare (optimize (speed 3) (safety 0))
           (type boolean is-first-time-call?)
           (type waffetensor out x y)
	   (ignore is-first-time-call? out))
  ; Todo Optimize 

  (warranty x)
  (warranty y)
  
  (value x)
  (value y)
  
  (let* ((dims (mat-dimensions (data x)))
	 (dims (if (and (= 1 (the fixnum (car (last dims))))
			(= 3 (length dims)))
		   (butlast dims)
		   dims))
	 (x1 (value x))
	 (dims (case (data y)
		 (1 `(,@(list (car dims)) 1))
		 (0 `(1 ,@(cdr dims)))
		 (T (error "Sum only supports a 2d matrix")))))
    (let ((o (mgl-mat:make-mat dims :initial-element 0.0)))
      (mgl-mat:sum! x1 o :axis (data y) :beta 0.0)
      (mgl-mat:reshape! o dims)
      (if (equal dims `(1 1))
	  (mat-as-scalar o)
	  o))))

(defun mean-tensor (is-first-time-call? out x y)
  (declare (optimize (speed 3) (safety 1))
           (type boolean is-first-time-call?)
           (type waffetensor out x y)
	   (ignore is-first-time-call? out))

  (warranty x)
  (warranty y)

  (value x)
  (value y)

  (let* ((dims (mgl-mat:mat-dimensions (value x)))
	 (dims (if (and (= 1 (the fixnum (car (last dims))))
			(= 3 (length dims)))
		   (butlast dims)
		   dims))
	 (x1 (mgl-mat:reshape! (mgl-mat:copy-mat (value x)) dims))
	 (dims (case (data y)
		 (1 `(,@(list (car dims)) 1))
		 (0 `(1 ,@(cdr dims)))
		 (T (error "Sum only supports a 2d matrix")))))
    (let ((o (mgl-mat:make-mat dims :initial-element 0))
	  (s (nth (value y) (mgl-mat:mat-dimensions (value x)))))
      (mgl-mat:sum! x1 o :axis (data y) :beta 1)
      (mgl-mat:scal! (/ 1 (the integer s)) x1)
      (mgl-mat:reshape! o dims)
      (if (equal dims `(1 1))
	  (mgl-mat:mref o 0 0)
	  o))))

;(declaim (ftype (function (boolean waffetensor waffetensor waffetensor) mgl-mat:mat) reshape-tensor))
(defun reshape-tensor (enable-optimize out x y &key (output nil) (overwrite nil))
  (declare (optimize (speed 3) (safety 1))
	   (type boolean enable-optimize)
	   (type waffetensor out x y))
  (let ((x1 (cond
	      ((not (null output))
	       output)
	      (overwrite
	       (value x))
	      ((cl-waffe::waffetensor-is-next-destruct? x)
	       (value x))
	      (T
	       (decide-out-buffer out (value x) enable-optimize t)))))
    (mgl-mat:reshape! x1 (data y))
    x1))

(define-lisp-kernel (bernoulli-lisp)
    ((mask :mat :io)
     (start-mask fixnum)
     (n fixnum)
     (p single-float))
  (loop for mi fixnum upfrom start-mask below (the fixnum (+ start-mask n))
	do (locally (declare (optimize (safety 1)))
	       (cond ((< (aref mask mi) p)
		  (setf (aref mask mi) 0.0))
		 (t
		  (setf (aref mask mi) 1.0))))))

(define-with-typevar bernoulli-tensor u (enable-optimize out return-tensor rate)
  (declare (type boolean enable-optimize)
	   (type waffetensor return-tensor rate))
  (let ((o (decide-out-buffer out return-tensor enable-optimize nil)))
    (mgl-mat:uniform-random! o)
    (bernoulli-lisp
     o
     (mat-displacement o)
     (mat-size o)
     (coerce (data rate) (quote u)))))


(define-lisp-kernel (embedding-forward-lisp)
    ((out :mat :output)
     (x :mat :input)
     (weights :mat :input)
     (n mgl-mat::index)
     (pad-idx fixnum)
     (embedding-dim fixnum))
  (loop for xi fixnum upfrom 0 below n
	do (if (= pad-idx (the fixnum (round (aref x xi))))
	      nil
	      (loop for ei fixnum upfrom 0 below embedding-dim
		    do (setf (aref out (the fixnum
					    (+
					     (the fixnum
						  (* embedding-dim xi))
					     ei)))
			     (aref weights
				   (the fixnum
					(+ ei
					   (the fixnum
						(* (the fixnum
							(round (aref x xi)))
						   embedding-dim))))))))))

(defun embedding-forward (enable-optimize x weights pad-idx)
  "(with-searching-calc-node :embedding-forward x weights pad-idx) -> embeddings"
  (declare (type boolean enable-optimize)
	   (type waffetensor x weights pad-idx)
	   (ignore enable-optimize))
  (let* ((batch-size (mat-dimension (data x) 0))
	 (total-size (mat-dimension (data x) 1))
	 (out (mgl-mat:make-mat `(,batch-size ,total-size ,(!shape weights 1))
				:initial-element 0.0)))
					; there's no cuda ver...
    (embedding-forward-lisp
     out
     (data x)
     (data weights)
     (mat-size (data x))
     (data pad-idx)
     (!shape weights 1))
    out))

(define-lisp-kernel (embedding-backward-lisp)
    ((out :mat :io)
     (x :mat)
     (dy :mat)
     (n fixnum)
     (pad-idx fixnum)
     (embedding-dim fixnum))
  (loop for xi of-type fixnum upfrom 0 below n
	do (cond
	     ((= pad-idx (the fixnum (round (aref x xi))))
	      nil)
	     (T
	      (loop for ei of-type fixnum upfrom 0 below embedding-dim
		    do (setf (aref out
				   (+ ei
				      (the fixnum
					   (* embedding-dim
					      (the fixnum (round (aref x xi)))))))
			     (+ (aref out
				      (+ ei
					 (the fixnum
					      (* embedding-dim
						 (the fixnum (round (aref x xi)))))))
				(aref dy (+ ei
					    (the fixnum
						 (* xi
						    (the fixnum embedding-dim))))))))))))

(defun embedding-backward (enable-optimize x dy weights pad-idx)
  (declare (type boolean enable-optimize)
	   (type waffetensor dy weights)
	   (ignore enable-optimize))
  (let ((out (mgl-mat:make-mat (!shape weights))))
    (embedding-backward-lisp
     out
     (data x)
     (data dy)
     (mgl-mat:mat-size (data x))
     (data pad-idx)
     (!shape dy 2))
    out))
#|
(declaim (ftype
	  (function
	   (keyword
	    boolean
	    waffetensor
	    waffetensor
	    cons
	    &key
	    (output (or null waffetensor))
	    (overwrite boolean))
	   (or mgl-mat:mat cl-waffe:waffedatatype))
	  dispatch-kernel))|#
(defun dispatch-kernel (function is-first-time-call? destructable-tensor destructable-tensor1 args &key (output nil) (overwrite nil))
  (declare (optimize (speed 3) (safety 0))
	   (type keyword function)
	   (type boolean is-first-time-call?)
	   (type waffetensor destructable-tensor)
	   (type cons args))
  (case function
    (:add     (kernel-add
	       is-first-time-call?
	       destructable-tensor
	       destructable-tensor1
	       :output output
	       :overwrite overwrite))
    (:sub     (kernel-sub
	       is-first-time-call?
	       destructable-tensor
	       destructable-tensor1
	       :output output
	       :overwrite overwrite))
    (:mul     (kernel-mul
	       is-first-time-call?
	       destructable-tensor
	       destructable-tensor1
	       :output output
	       :overwrite overwrite))
    (:div     (kernel-div
	       is-first-time-call?
	       destructable-tensor
	       destructable-tensor1
	       :output output
	       :overwrite overwrite))
    (:dot     (dot-tensor is-first-time-call? destructable-tensor (car args) (second args)))
    (:matmul  (matmul-tensor is-first-time-call? destructable-tensor (car args) (second args)))    
    (:log     (kernel-log is-first-time-call? destructable-tensor
			  :output output
			  :overwrite overwrite))
    (:exp     (kernel-exp is-first-time-call? destructable-tensor
			  :output output
			  :overwrite overwrite))
    (:sqrt    (kernel-sqrt is-first-time-call? destructable-tensor
			   :output output
			   :overwrite overwrite))
    (:sin     (kernel-sin is-first-time-call? destructable-tensor
			  :output output
			  :overwrite overwrite))
    (:cos     (kernel-cos is-first-time-call? destructable-tensor
			  :output output
			  :overwrite overwrite))
    (:tan     (kernel-tan is-first-time-call? destructable-tensor
			  :output output
			  :overwrite overwrite))
    (:asin    (kernel-asin is-first-time-call? destructable-tensor
			   :output output
			   :overwrite overwrite))
    (:acos    (kernel-acos is-first-time-call? destructable-tensor
			   :output output
			   :overwrite overwrite))
    (:atan    (kernel-atan is-first-time-call? destructable-tensor
			   :output output
			   :overwrite overwrite))
    
    (:sinh    (kernel-sinh is-first-time-call? destructable-tensor
			   :output output
			   :overwrite overwrite))
    (:cosh    (kernel-cosh is-first-time-call? destructable-tensor
			   :output output
			   :overwrite overwrite))
    (:tanh    (kernel-tanh is-first-time-call? destructable-tensor
			   :output output
			   :overwrite overwrite))
    
    (:asinh   (kernel-asinh is-first-time-call? destructable-tensor
			    :output output
			    :overwrite overwrite))
    (:acosh   (kernel-acosh is-first-time-call? destructable-tensor
			    :output output
			    :overwrite overwrite))
    (:atanh   (kernel-atanh is-first-time-call? destructable-tensor
			    :output output
			    :overwrite overwrite))
    
    (:pow     (kernel-pow
	       is-first-time-call?
	       destructable-tensor
	       destructable-tensor1
	       :output output
	       :overwrite overwrite))
    (:sum     (sum-tensor is-first-time-call? destructable-tensor (car args) (second args)))
    (:mean    (mean-tensor is-first-time-call? destructable-tensor (car args) (second args)))
    (:reshape (reshape-tensor is-first-time-call? destructable-tensor (car args) (second args) :output output
	       :overwrite overwrite))
    (:<       (compare-tensor is-first-time-call? destructable-tensor (car args) (second args)))
    (:repeat  (mgl-repeat (data (car args)) (data (third args)) :axis (data (second args))))
    (:bernoulli (bernoulli-tensor is-first-time-call? destructable-tensor (car args) (second args)))
    (:transpose (lazy-eval-transpose
		 (data (car args))
		 nil))
    (:embedding-forward (embedding-forward nil (car args)
					   (second args)
					   (third args)))
    (:embedding-backward (embedding-backward nil
					     (car args)
					     (second args)
					     (third args)
					     (fourth args)))
    (T (error "~a is not yet implemented" function))))

