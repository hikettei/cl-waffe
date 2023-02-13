
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
  :backward ((dy) (list (!mul (self yi) dy)
			(!mul (self xi) dy))))

(defnode DivTensor nil
  :optimize t
  :parameters ((xi T) (yi T))
  :forward ((x y)
	    (unless (= (data x) 1) (error "!div-old: x must be 1"))
            (save-for-backward xi x)
	    (save-for-backward yi y)
	    (with-searching-calc-node :div x y))
  :backward ((dy) (list (!div dy (self yi))
			(!div (!mul (!mul (self xi) dy) -1)
			      (!pow (self yi) 2)))))

(defnode PowTensor nil
  :optimize t
  :parameters ((xi T) (yi T))
  :forward ((x1 y1)
	    (save-for-backward xi x1)
	    (save-for-backward yi y1)
	    (with-searching-calc-node :pow x1 y1))
  :backward ((dy)
	     (list (!mul (!mul dy (self yi))
			 (!pow (self xi) (- (the single-float (data (self yi))) 1)))
		   (!mul (!mul
			     (!log (self xi))
			     (!pow (self xi) (self yi)))
			 dy))))

(defnode SqrtTensor nil
  :optimize t
  :parameters ((xi T))
  :forward ((x1) (save-for-backward xi x1)
		 (with-searching-calc-node :sqrt x1))
  :backward ((dy)
	     (list (!div dy (!mul (!sqrt (self xi)) 2)))))

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
  :forward ((x) ; only for 2d
	    (setf (self total-len) (/ (!size x)))
	    (setf (self shape) (!shape x))
	    (!sum (!sum x 1) 0))
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
	     (list (!mul (!exp (self xi)) dy))))

(defnode MatMulTensor ()
  :optimize t
  :parameters ((xi nil) (yi nil))
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
       (declare (optimize (speed 3) (safety 0)))
       (let* ((,tensor (if *no-grad* ,place ,node-object)))
	,@body))))

(defope !add (AddTensor) node (x y)
    "Adds x and y.

In the case when x or y is not a tensor, automatically creates a new tensor.

It supports:

@begin(enum)
@item(Broadcasting shapes)
@item(JIT)
@end(enum)

@begin(section)
@title(Examples)
@begin[lang=lisp](code)

(setq a (!randn `(3 3)))
(setq b (!randn `(3 3)))
(setq c (!randn `(3 1)))

(!add 1 1)
;=> Const(2)

(!add (const 1) (const 1))
;=> Const(2)

(!add a b)
;#Const(((3.418... 1.974... 0.177...)
;                 ...
;        (-1.30... 0.987... 1.917...)) :mgl t :shape (3 3))

(!add a c)
;#Const(((1.426... 2.129... 1.050...)
;                 ...
;        (-0.64... 0.269... 0.303...)) :mgl t :shape (3 3))

@end[lang=lisp](code)
@end(section)"
  (call node (assure-tensor x) (assure-tensor y)))
    
(defope !sub (SubTensor) node (x y)
    "Subtract x by y.

In the case when x or y is not a tensor, automatically creates a new tensor.

It supports:

@begin(enum)
@item(Broadcasting shapes)
@item(JIT)
@end(enum)

@begin(section)
@title(Examples)
@begin[lang=lisp](code)

(setq a (!randn `(3 3)))
(setq b (!randn `(3 3)))
(setq c (!randn `(3 1)))

(!sub 1 1)
;=> Const(0)

(!sub (const 1) (const 1))
;=> Const(0)

(!sub a b)
;#Const(((-0.86... 1.413... 1.139...)
;                 ...
;        (0.017... -0.44... -1.31...)) :mgl t :shape (3 3))

(!sub a c)
;#Const(((1.128... 1.258... 0.267...)
;                 ...
;        (-0.64... 0.269... 0.303...)) :mgl t :shape (3 3))

@end[lang=lisp](code)
@end(section)
"

  (call node (assure-tensor x) (assure-tensor y)))

(defope !mul (MulTensor) node (x y)
    "Multiply x and y with element-wise.

In the case when x or y is not a tensor, automatically creates a new tensor.

It supports:

@begin(enum)
@item(Broadcasting shapes)
@item(JIT)
@end(enum)

@begin(section)
@title(Examples)
@begin[lang=lisp](code)

(setq a (!randn `(3 3)))
(setq b (!randn `(3 3)))
(setq c (!randn `(3 1)))

(!mul 1 1)
;=> Const(1)

(!mul (const 1) (const 1))
;=> Const(1)

(!mul a b)
;#Const(((2.734... 0.475... -0.31...)        
;                 ...
;        (0.426... 0.193... 0.490...)) :mgl t :shape (3 3))

(!mul a c)
;#Const(((2.734... 0.475... -0.31...)        
;                 ...
;        (0.426... 0.193... 0.490...)) :mgl t :shape (3 3))

@end[lang=lisp](code)
@end(section)
"
  (call node (assure-tensor x) (assure-tensor y)))

(defope !div-old (DivTensor) node (x y)
   "1/x"
  ; (unless (= x 1) (error "!div-old: x must be 1"))
  ; x must be 1, cl-waffe.backends.mgl:div has some problems?...
  (call node (assure-tensor x) (assure-tensor y)))

; its much faster
(defun !div (x y)
  "Divides x by y.

In the case when x or y is not a tensor, automatically creates a new tensor.

It supports:

@begin(enum)
@item(Broadcasting shapes)
@item(JIT)
@end(enum)

@begin(section)
@title(Examples)
@begin[lang=lisp](code)

(setq a (!randn `(3 3)))
(setq b (!ones `(3 3)))
(setq c (!ones `(3 1)))

(!div 2 1)
;=> Const(2)

(!div (const 2) (const 1))
;=> Const(2)

(!div a b)
;#Const(((1.734... 0.475... -0.31...)        
;                 ...
;        (0.426... 0.193... 0.490...)) :mgl t :shape (3 3))

(!div a c)
;#Const(((2.734... 0.475... -0.31...)        
;                 ...
;        (0.426... 0.193... 0.490...)) :mgl t :shape (3 3))

@end[lang=lisp](code)
@end(section)
"
  (!mul x (!div-old 1 y)))
  
(defope !dot (DotProductTensor) node (x y)
    "Computes the dot product of x and y where x and y are 1d Tensor.

🗒Note: Unlike Numpy's dot, !dot only supports for 1d tensors with the same number of elements and the tensor of which dims is larger than 1, regarded as 1d tensors.

@begin(section)
@title(Example)
@begin[lang=lisp](code)

(setq a (!randn `(10)))
(setq b (!randn `(10)))

(!dot a b)
;=> #Const(1.0842022e-19)
@end[lang=lisp](code)
@end(section)
"
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
  "Sum up x where x is a cl-waffe tensor.

For nd tensors...
@begin(deflist)
@def(1D)
@term(unsqueeze x with 1, and call !sum again.)
@def(2D and more.)
@term(Sum up all elements of X)
@end(deflist)

@begin(section)
@title(arguments)

@begin(deflist)
@def(axis)
@term(a dimension to reduce)
@def(keepdims)
@term(When t, the returning tensor is repeated with @cl:param(axis))
@end(deflist)

@end(section)

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!randn `(10)))
(!sum a)
;=>#Const(4.74653)

(setq a (!randn `(10 10)))
(!sum a)
;=>#Const(1.5428619)

(!sum a 0)
;=>#Const(((-2.07... 0.463... ~ 1.778... 1.695...)) :mgl t :shape (1 10))

(!sum a 1)
;#Const(((0.967...)        
;                 ...
;        (2.774...)) :mgl t :shape (10 1))

(!sum a 0 t)
;#Const(((-2.07... 0.463... ~ 1.778... 1.695...)        
;                 ...
;        (-2.07... 0.463... ~ 1.778... 1.695...)) :mgl t :shape (10 10))
@end[lang=lisp](code)
@end(section)
"

  (declare (type (or null fixnum) axis)
	   (type boolean keepdims)
	   (type waffetensor x))
  (case (!dims x)
    (0 (error "!sum: the tensor given is a number"))
    (1 (!sum-2d (!unsqueeze x 1) axis keepdims))
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
  "The usage is the same as !sum.

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!ones '(10 10)))
;#Const(((1.0 1.0 ~ 1.0 1.0)        
;                 ...
;        (1.0 1.0 ~ 1.0 1.0)) :mgl t :shape (10 10))
(!mean a)
;=>Const(1.0)
@end[lang=lisp](code)
@end(section)"
  (if (null axis)
      (!div (!sum x axis keepdims) (apply #'* (!shape x)))
      (!div (!sum x axis keepdims) (!shape x axis))))

(defope !pow (PowTensor) node (x n)
    "Takes the power of each element in @cl:param(x) with n, returning a new sysconst.

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!ones `(10 10)))
(!pow a 3)
;#Const(((1.0 1.0 ~ 1.0 1.0)        
;                 ...
;        (1.0 1.0 ~ 1.0 1.0)) :mgl t :shape (10 10))
@end[lang=lisp](code)
@end(section)"
  (call node (assure-tensor x) (assure-tensor n)))

(defope !sqrt (SqrtTensor) node (x)
    "Takes the power of eachelement in @cl:param(x) with 1/2, creating new sysconst and nodes.

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!ones `(10 10)))
(!sqrt a 3)
;#Const(((1.0 1.0 ~ 1.0 1.0)
;                 ...
;        (1.0 1.0 ~ 1.0 1.0)) :mgl t :shape (10 10))
@end[lang=lisp](code)
@end(section)"
  (call node (assure-tensor x)))

(defope !log (LogTensor) node (x)
    "Returns a new tensor with the natural logarithm of the elements of input.

yi = log(e xi)

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!ones '(10 10)))
(!log a)
;#Const(((0.0 0.0 ~ 0.0 0.0)        
;                 ...
;        (0.0 0.0 ~ 0.0 0.0)) :mgl t :shape (10 10))
@end[lang=lisp](code)
@end(section)"
  (call node (assure-tensor x)))

(defun !reshape (x dim)
  "Return a new sysconst with changing its shape. x won't be modified.

If dims has the element of @cl:param(t), t is automatically inferred from the remaining dimensions and the number of elements in dim. (count t dim) must be 1 (Todo: Fix).

The total size of tensor must not be changed before or after the call to reshape.

See also: nil

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!randn `(10 10 10)))
(!reshape a '(1 10 100))
;#Const((((0.454... 0.277... ~ 0.536... 0.135...)         
;                   ...
;         (0.857... 0.714... ~ 0.169... 0.279...))) :mgl t :shape (1 10 100))

(!reshape a '(1 1 t))
;#Const((((0.454... 0.277... ~ 0.169... 0.279...))) :mgl t :shape (1 1 1000))
@end[lang=lisp](code)
@end(section)"
  (declare (type cons dim))
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
  "Repeats @cl:param(x) along specified @cl:param(axis) by @cl:param(repeats), creating new sysconst.

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!randn '(1 3 3)))
;#Const((((0.333... 0.914... 0.260...)         
;                   ...
;         (0.611... 0.110... 0.113...))) :mgl t :shape (1 3 3))
(!repeats a 0 3)
;#Const((((0.333... 0.914... 0.260...)         
;                   ...
;         (0.611... 0.110... 0.113...))        
;                 ...
;        ((0.333... 0.914... 0.260...)         
;                   ...
;         (0.611... 0.110... 0.113...))) :mgl t :shape (3 3 3))
@end[lang=lisp](code)
@end(section)"
  (declare (type waffetensor x)
	   (type fixnum axis repeats))
  (call (RepeatTensor (assure-tensor axis) (assure-tensor repeats)) (assure-tensor x)))

(defun !transpose (x &optional result)
  "Transpose x where x is a 2d tensor.

Transposed x is lazy evaluated until called by !matmul.

Todo: implement 3d, 4d version...

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!randn `(3 5)))
(setq a (!transpose a))
;#Const(#<FUNCTION (LABELS CL-WAFFE.BACKENDS.MGL::LAZYTRANSPOSE :IN CL-WAFFE.BACKENDS.MGL::LAZY-EVAL-TRANSPOSE) {10038CBADB}>)

(!matmul a (!randn '(3 5)))
;#Const(((0.653... 0.400... 0.471... 0.705... 0.623...)        
;                 ...
;        (1.220... 0.760... 0.975... 1.360... 1.029...)) :mgl t :shape (5 5))
@end[lang=lisp](code)
@end(section)"
  (call (TransposeTensor (assure-tensor result)) (assure-tensor x)))

(defope !matmul (MatmulTensor) node (x y)
    "Multiplying matrices @cl:param(x) and @cl:param(y).

!matmul has many behaviours depends on the dimensionality of the tensors as follows:

@begin(deflist)
@def(x and y are 1D)
@begin(term)
The dot-product is returned.
@begin[lang=lisp](code)
(setq a (!randn `(10)))
(setq b (!randn `(10)))
(!matmul a b)
;=>#Const(-2.0)
@end[lang=lisp](code)
@end(term)

@def(x and y are both 2D)
@begin(term)
The matrix-matrix product is returned.
@begin[lang=lisp](code)
(setq a (!randn `(3 10)))
(setq b (!randn `(10 3)))
(!matmul a b)
;#Const(((2.309... 2.223... 3.630...)        
;                 ...
;        (2.334... 2.850... 3.678...)) :mgl t :shape (3 3))
@end[lang=lisp](code)
@end(term)

@def(x is 2D and y is 3D.)
@begin(term)
The matrix and y's each matrix are multiplied and is returned.
@begin[lang=lisp](code)
(setq a (!randn `(3 10)))
(setq b (!randn `(5 10 3)))

(!matmul a b)
;(!aref b 0) ~ (!aref b 4) is multiplied with a

;#Const((((3.257... 2.731... 1.670...)         
;                   ...
;         (2.523... 2.251... 1.276...))        
;                 ...
;        ((2.610... 2.764... 2.415...)         
;                   ...
;         (2.080... 2.204... 1.751...))) :mgl t :shape (5 3 3))
@end[lang=lisp](code)
@end(term)

@def(x is 3D and y is 2D.)
@begin(term)
The matrix and x's each matrix are multiplied and is returned.
@begin[lang=lisp](code)
(setq a (!randn `(5 3 10)))
(setq b (!randn `(10 3)))

(!matmul a b)
;(!aref a 0) ~ (!aref a 4) is multiplied with b
;#Const((((2.309... 2.204... 1.556...)         
;                   ...
;         (3.746... 3.869... 3.091...))        
;                 ...
;        ((3.260... 3.200... 2.847...)         
;                   ...
;         (3.008... 2.186... 2.376...))) :mgl t :shape (5 3 3))
@end[lang=lisp](code)
@end(term)

@def(For more...)
@term(More will be added (e.g.: 1d and 2d, for larger than 4d ...))

@end(deflist)"
  (cond
    ((and (= (the fixnum (!dims x)) 1)
	  (= (the fixnum (!dims y)) 1))
     (!dot x y))
    (T (call node (assure-tensor x) (assure-tensor y)))))
	
(defun !unsqueeze (x &optional (dim 0))
  "Unsqueeze Todo: write docs


@begin(section)
@title(Example)
@begin[lang=lisp](code)

@end[lang=lisp](code)
@end(section)"
  ; display error when (!dims x) >= dim
  (let ((s (!shape x)))
    (case dim
      (0  (setq s `(1 ,@s)))
      (-1 (push 1 (cdr (nthcdr (1- (length s)) s))))
      (T  (push 1 (cdr (nthcdr (1- dim) s)))))
    (!reshape x s)))

(defun !squeeze (x &optional (dim nil))
  "Squeeze todo: write docs


@begin(section)
@title(Example)
@begin[lang=lisp](code)

@end[lang=lisp](code)
@end(section)"
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
    "Exp x, creating new sysconst and nodes.

@begin(section)
@title(Example)
@begin[lang=lisp](code)

@end[lang=lisp](code)
@end(section)"
  
  (call node (assure-tensor x)))

(defun !sin () "Todo")
(defun !cos () "Todo")
(defun !tan () "Todo")

(defun !asin () "Todo")
(defun !acos () "Todo")
(defun !atan () "Todo")

(defun !sinh () "Todo")
(defun !cosh () "Todo")

(defun !argmax () "Todo")
(defun !argmin () "Todo")

(defun !abs () "Todo")

(defun != () "Todo")
(defun !<= () "Todo")
(defun !>= () "Todo")

(defun !einsum () "Todo")
(defun !ravel () "Todo")
(defun !flatten () "Todo")

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

