
(in-package :cl-waffe)

(defgeneric assure-tensor (x))

(defmethod assure-tensor ((x waffetensor)) x)
(defmethod assure-tensor ((x fixnum))   (const x))
(defmethod assure-tensor ((x float))    (const x))
(defmethod assure-tensor ((x null))     (const x))
(defmethod assure-tensor ((x cons))     (const x))
(defmethod assure-tensor ((x function)) (const x))
(defmethod assure-tensor ((x ratio))    (const x))
(defmethod assure-tensor ((x mgl-mat:mat)) (const x))

(defparameter *instruction-map* (alist-hash-table `((:+= . :add)
						    (:-= . :sub)
						    (:*= . :mul)
						    (:/= . :div)
						    (:log . :log)
						    (:exp . :exp)
						    (:^= . :pow)
					            (:sqrt . :sqrt)
				                    (:tanh . :tanh)
				        	    (:reshape . :reshape)
						    (:< . :<)
						    (:bernoulli . :bernoulli))))
(declaim (inline !div !transpose))

(defnode AddTensor nil
  :optimize t
  :parameters nil
  :forward  ((x y)
	     (with-searching-calc-node :add x y))
  :backward ((dy) (list dy dy)))

(defnode SubTensor nil
  :optimize t
  :parameters ()
  :forward ((x y) (with-searching-calc-node :sub x y))
  :backward ((dy) (list dy (!mul dy (const -1)))))

(defnode MulTensor nil
  :optimize t
  :parameters ((xi T) (yi T))
  :forward ((x y)
	    (save-for-backward xi x)
	    (save-for-backward yi y)
	    (with-searching-calc-node :mul x y))
  :backward ((dy) (list (!modify (self yi) :*= dy)
			(!modify (self xi) :*= dy))))

(defnode DivTensor nil
  :optimize t
  :parameters ((xi T) (yi T))
  :forward ((x y)
	    (unless (= (data x) 1) (error "!div-old: x must be 1"))
            (save-for-backward xi x)
	    (save-for-backward yi y)
	    (with-searching-calc-node :div x y))
  :backward ((dy) (list (!div dy (self yi))
			(!div (!modify (!modify (self xi) :*= dy) :*= -1)
			      (!modify (self yi) :^= 2)))))

(defnode PowTensor nil
  :optimize t
  :parameters ((xi T) (yi T))
  :forward ((x1 y1)
	    (save-for-backward xi x1)
	    (save-for-backward yi y1)
	    (with-searching-calc-node :pow x1 y1))
  :backward ((dy)
	     (list (!modify (!mul dy (self yi)) :*= (!pow (self xi) (- (the single-float (data (self yi))) 1)))
		   (!modify (!modify
			     (!log (self xi)) :*=
			     (!modify (self xi) :^= (self yi)))
			    :*= dy))))

(defnode SqrtTensor nil
  :optimize t
  :parameters ((xi T))
  :forward ((x1) (save-for-backward xi x1)
		 (with-searching-calc-node :sqrt x1))
  :backward ((dy)
	     (list (!div dy (!modify (!modify (self xi) :sqrt) :*= 2)))))

(defnode LogTensor nil
  :optimize t
  :parameters ((x1 T))
  :forward ((x1) (save-for-backward x1 x1)
		 (with-searching-calc-node :log x1))
  :backward ((dy) (list (!div dy (self x1)))))

(defnode ReshapeTensor (shape)
  :optimize t
  :parameters ((prev-shape T) (shape shape))
  :forward ((x) (setf (self prev-shape) (!shape x))
		(with-searching-calc-node :reshape x (self shape)))
  :backward ((dy)
	     (print dy)
	     (list (!reshape dy (self prev-shape)))))

(defnode DotProductTensor nil
  :optimize t
  :parameters ((xi T) (yi T))
  :forward ((x1 x2) ; only supports 2d and 2d arrays
		    (save-for-backward xi x1)
		    (save-for-backward yi x2)
		    (with-searching-calc-node :dot x1 x2))
  :backward ((dy)
	       (list (!dot dy (!transpose (self yi)))
		     (!dot (!transpose (self xi)) dy))))

(defnode TransposeTensor (shape)
  :optimize t
  :parameters ((prev-shape T) (shape shape))
  :forward ((x)
	    (setf (self prev-shape) (assure-tensor (!shape x)))
	    (with-searching-calc-node :transpose x (self shape)))
  :backward ((d1)
	     (list (const (mgl-mat:transpose (data d1))))))

(defnode MeanTensor (axis)
  :optimize t
  :parameters ((axis axis) (repeats T))
  :forward ((x)
	    (setf (self repeats) (assure-tensor (!shape x (self axis))))
	    (with-searching-calc-node :mean x (self axis)))
  :backward ((dy) (list (!repeats dy (self axis) (self repeats)))))

(defnode SumTensor (axis)
  :optimize t
  :parameters ((axis axis) (repeats T))
  :forward ((x)
	    (setf (self repeats) (assure-tensor (!shape x (self axis))))
	    (with-searching-calc-node :sum x (self axis)))
  :backward ((dy)
	     (list (!div (!repeats dy (self axis) (self repeats))
			 (self repeats)))))

(defnode SumUpTensor ()
  :parameters ((total-len) (shape))
  :forward ((x)
	    (setf (self total-len) (/ (!size x)))
	    (setf (self shape) (!shape x))
	    (with-kernel-case x out
	      :mgl ((* -1 (asum out))) ; asum >= 0
	      :mgl-cuda nil))
  :backward ((dy)
	     (list (sysconst (scal! (self total-len)
				    (make-mat (self shape)
					      :initial-element (data dy)))))))

(defnode RepeatTensor (axis repeats)
  :optimize t
  :parameters ((axis axis) (repeats repeats))
  :forward ((x)
	    (with-searching-calc-node :repeat x (self axis) (self repeats)))
  :backward ((dy) (list (!sum dy (self axis)))))

(defnode ExpTensor ()
  :optimize t
  :parameters ((xi T))
  :forward ((x) (save-for-backward xi x)
		(with-searching-calc-node :exp x))
  :backward ((dy)
	     (list (!modify (!exp (self xi)) :*= dy))))

(defnode MatMulTensor ()
  :optimize t
  :parameters ((xi T) (yi T))
  :forward ((x y) (save-for-backward xi x)
		  (save-for-backward yi y)
		  (with-searching-calc-node :matmul x y))
  :backward ((dy)
	     (list (!matmul dy (!transpose (self yi)))
		   (!matmul (!transpose (self xi)) dy))))

(defmacro defope (name node-object tensor args &optional (doc "") &body body)
  (let ((place node-object))
    `(defun ,name ,args
       ,doc
       (let* ((,tensor (if *no-grad* ,place ,node-object)))
	,@body))))

(defope !add (AddTensor) node (x y)
    "Add x y, creating new sysconst and nodes.

As a modify: (!modify x :+= y)"
  (call node (assure-tensor x) (assure-tensor y)))
    
(defope !sub (SubTensor) node (x y)
    "Subtract x by y, creating new sysconst and nodes.

As a modify: (!modify x :-= y)"
  (call node (assure-tensor x) (assure-tensor y)))

(defope !mul (MulTensor) node (x y)
    "Mul x y, creating new sysconst and nodes

As a modify: (!modify x :*= y)"
  (call node (assure-tensor x) (assure-tensor y)))

(defope !div-old (DivTensor) node (x y)
   "1/x"
  ; (unless (= x 1) (error "!div-old: x must be 1"))
  ; x must be 1, cl-waffe.backends.mgl:div has some problems?...
  (call node (assure-tensor x) (assure-tensor y)))

; its much faster
(defun !div (x y)
  "Div x y, creating new sysconst and nodes.

As a modify: (!modify x :/= y)"
  (!mul x (!div-old 1 y)))
  
(defope !dot (DotProductTensor) node (x y)
  "Dot product of x and y, creating new sysconst and nodes"
  ; Todo: dot excepts 1d tensor
  (call node (assure-tensor x) (assure-tensor y)))

(defun !sum-2d (x &optional (axis nil) (keepdims nil))
  (if (null axis)
      (call (SumUpTensor) (assure-tensor x))
      (let ((nrepeat (!shape x axis))
	    (result (call (SumTensor (assure-tensor axis)) (assure-tensor x))))
	(if keepdims
	    (!repeats result axis nrepeat)
	    result))))

(defun !sum (x &optional (axis nil) (keepdims nil))
  "Sum up X where x is a tensor (the size of dims doesn't matter.)

Todo: Write docs and examples"
  (case (!dims x)
    (0 (error "!sum: the tensor given is a number"))
    (1 (!sum-2d x axis keepdims))
    (2 (!sum-2d x axis keepdims))
    (T
     (if (null axis)
	 (let ((result (!sum (!squeeze (!aref x 0)))))
	   (loop for i upfrom 1 below (!shape x 0)
		 do (setq result (!add result (!sum (!aref x i)))))
	   result)
	 (let* ((dims (!shape x axis))
	       ; Note: keepdims is ignored. And May need exclusive kernel for it because its too slow when forward and backward.

		(sum-dims #'(lambda (n) (loop for i upfrom 0 below (!dims x)
	 				      collect (if (= i axis)
							  n
							  t))))
		(result (!zeros (!shape (apply #'!aref x (funcall sum-dims 0))))))
	   (dotimes (i dims)
	     (setq result (!add result (apply #'!aref x (funcall sum-dims i)))))
	   result)))))

(defun !mean (x &optional (axis nil) (keepdims nil))
  "Mean of tensor, todo: write docs and examples"
  (if (null axis)
      (let ((axis-size (!dims x))
	    (result x))
	(dotimes (i axis-size)
	  (setq result (!mean result (1- (- axis-size i)))))
	result)
      (let ((nrepeat (!shape x axis))
	    (result (call (MeanTensor (assure-tensor axis)) (assure-tensor x))))
	(if keepdims
	    (!repeats result axis nrepeat)
	    result))))

(defope !pow (PowTensor) node (x n)
    "Pow x n, creating new sysconst and nodes.

As a modify: (!modify x :^= y)"
  (call node (assure-tensor x) (assure-tensor n)))

(defope !sqrt (SqrtTensor) node (x)
    "Sqrt x, creating new sysconst and nodes.

As a modify: (!modify x :sqrt y)"
  (call node (assure-tensor x)))

(defope !log (LogTensor) node (x)
  "Log x, creating new sysconst and nodes."
  (call node (assure-tensor x)))

(defun !reshape (x dim)
  "(!reshape x dim) ,if dim has t, t is automatically predicted."
  (if (find t dim)
      (progn
	(unless (= (count t dim) 1)
	  (error "cl-waffe:!reshape: auto inference of shape supports only when (count t dim) = 1"))
	(let* ((dim (copy-list dim))
	       (total-size  (apply #'* (!shape x)))
	       (remain-size (apply #'* (map 'list (lambda (x)
						    (if (eql x T)
							1
							x))
					    dim)))
	       (predicted-dim (/ total-size remain-size)))
	  (setf (nth (position t dim) dim) predicted-dim)
	  (call (ReshapeTensor (assure-tensor dim)) (assure-tensor x))))
      (call (ReshapeTensor (assure-tensor dim)) (assure-tensor x))))

(defun !repeats (x axis repeats)
  "Repeat Todo: Write docs and example"
  (call (RepeatTensor (assure-tensor axis) (assure-tensor repeats)) (assure-tensor x)))

(defun !transpose (x &optional result)
  "Transpose

Note: the result of !transpose is lazy evaluated for speed.(Todo: Write details)"
  (call (TransposeTensor (assure-tensor result)) (assure-tensor x)))

(defope !matmul (MatmulTensor) node (x y)
    "Matmul

Todo: write docs and behaviour"
  (call node (assure-tensor x) (assure-tensor y)))
	
(defun !unsqueeze (x &optional (dim 0))
  "Unsqueeze Todo: write docs"
  ; display error when (!dims x) >= dim
  (let ((s (!shape x)))
    (case dim
      (0  (setq s `(1 ,@s)))
      (-1 (push 1 (cdr (nthcdr (1- (length s)) s))))
      (T  (push 1 (cdr (nthcdr (1- dim) s)))))
    (!reshape x s)))

(defun !squeeze (x &optional (dim nil))
  "Squeeze todo: write docs"
  (labels ((remove-nth (nth list)
	     (loop for i in list
		   for idx from 0
		   unless (= idx nth)
		     collect i)))
    (let ((s (!shape x)))
      (cond
	((null dim) (setq s (remove 1 s)))
	((eq dim 0) (setq s (if (= (car s) 1)
		       (cdr s)
		       s)))
	((eq dim -1) (setq s (if (= (car (last s)) 1)
			(butlast s)
			s)))
	(T (setq s (if (= (nth dim s) 1)
		       (remove-nth dim s)
		       s))))
      (!reshape x s))))

(defope !exp (ExpTensor) node (x)
  "Exp x, creating new sysconst and nodes."
  (call node (assure-tensor x)))

(declaim (ftype (function ((or mgl-mat:mat waffetensor) keyword &rest (or waffedatatype waffetensor)) waffetensor) !modify))
(defun !modify (target instruction &rest args)
  (declare (optimize (speed 3) (space 0) (safety 0)))
  "The function that allows destructively operations, always changing the target.

If you need mgl-mat-wise operations for speed and low memory, this is useful.

Directly Calling Mgl-mat Operations.

Please remain that it won't make backwards because of speed problems.(Todo: Fix)

Always return `target` tensor. target always changed, and args sometimes changed

Instruction is a symbol where described with modify:

Todo: Write more details.

Example:

@begin[lang=lisp](code)
(!modify x :+= y)
@end[lang=lisp](code)"
  (unless (gethash instruction *instruction-map*)
    (error "!modify: The instruction ~a is not found. please check the documentation" instruction))
  
  (with-optimized-operation
      (with-searching-calc-node-optim (gethash instruction *instruction-map*)
	(data (assure-tensor target))
	(assure-tensor target)
        (map 'list (lambda (x)
		     (declare (type (or waffetensor waffedatatype) x))
		     (the waffetensor (assure-tensor x)))
	     args))))

