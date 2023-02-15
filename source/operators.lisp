
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
	     (list (!transpose d1))))

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

(defnode SinTensor ()
  :optimize t
  :parameters ((xi nil))
  :forward ((x)
	    (save-for-backward xi x)
	    (with-searching-calc-node :sin x))
  :backward ((dy)
	     (list (!mul dy (!cos (self x))))))

(defnode CosTensor ()
  :optimize t
  :parameters ((xi nil))
  :forward ((x)
	    (save-for-backward xi x)
	    (with-searching-calc-node :cos x))
  :backward ((dy)
	     (list (!mul dy (!mul -1.0 (!sin (self x)))))))

(defnode TanTensor ()
  :optimize t
  :parameters ((xi nil))
  :forward ((x)
	    (save-for-backward xi x)
	    (with-searching-calc-node :tan x))
  :backward ((dy)
	     (list (!mul dy (!div 1 (!pow (!cos (self x)) 2))))))

(defnode HyperbolicSinTensor ()
  :optimize t
  :parameters ((xi nil))
  :forward ((x)
	    (save-for-backward xi x)
	    (with-searching-calc-node :sinh x))
  :backward ((dy)
	     (list (!mul dy (!cosh (self x))))))

(defnode HyperbolicCosTensor ()
  :optimize t
  :parameters ((xi nil))
  :forward ((x)
	    (save-for-backward xi x)
	    (with-searching-calc-node :cosh x))
  :backward ((dy)
	     (list (!mul dy (!sinh (self x))))))

(defnode AbsTensor ()
  :optimize t
  :parameters ((mask nil))
  :forward ((x)
	    (let ((mask (!where #'(lambda (x)
				    (declare (type single-float x))
				    (> x 0.0))
				x 1.0 -1.0)))
	      (save-for-backward mask x)
	      (!mul x mask)))
  :backward ((dy)
	     (list (!mul dy (self mask)))))

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
  "Returns a new tensor with a dimension of size one inserted at the specified position.

dim indicates the position, when dim=-1, it indicates a last dimension of @cl:param(x).

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!randn `(10 10)))
;#Const(((0.685... 0.827... ~ 0.076... 0.102...)        
;                 ...
;        (0.802... 0.571... ~ 0.207... 0.283...)) :mgl t :shape (10 10))
(!unsqueeze a)
;#Const((((0.685... 0.827... ~ 0.076... 0.102...)         
;                   ...
;         (0.802... 0.571... ~ 0.207... 0.283...))) :mgl t :shape (1 10 10))

(!unsqueeze a -1)
;#Const((((0.685...)         
;                   ...
;         (0.102...))        
;                 ...
;        ((0.802...)         
;                   ...
;         (0.283...))) :mgl t :shape (10 10 1))

(!unsqueeze a 2)
;#Const(((0.685... 0.827... ~ 0.076... 0.102...)        
;                 ...
;        (0.802... 0.571... ~ 0.207... 0.283...)) :mgl t :shape (10 10 1 1))
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
  "Returns a new tensor with a dimension of size one removed at the specified position.

When dim=nil or -1, the last position of dim will be removed.

If the specified position of a tensor isn't one, !squeeze is skipped.

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!randn `(10 1 10)))
;#Const((((0.928... 0.556... ~ 0.697... 0.973...))        
;                 ...
;        ((0.368... 0.995... ~ 0.589... 0.716...))) :mgl t :shape (10 1 10))

(!squeeze a 1)
;#Const(((0.928... 0.556... ~ 0.697... 0.973...)        
;                 ...
;        (0.368... 0.995... ~ 0.589... 0.716...)) :mgl t :shape (10 10))

(!squeeze a -1)
;#Const((((0.928... 0.556... ~ 0.697... 0.973...))        
;                 ...
;        ((0.368... 0.995... ~ 0.589... 0.716...))) :mgl t :shape (10 1 10))

(setq a (!randn `(10 10 1)))
;#Const(((0.991... 0.248... ~ 0.610... 0.289...)        
;                 ...
;        (0.593... 0.177... ~ 0.374... 0.668...)) :mgl t :shape (10 10))
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
    "Applying exp to each element of x, creating a new sysconst.

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!randn `(10 10)))
;#Const(((0.624... 0.807... ~ 0.500... 0.937...)        
;                 ...
;        (0.662... 0.299... ~ 0.761... 0.729...)) :mgl t :shape (10 10))
(!exp a)
;#Const(((1.866... 2.242... ~ 1.650... 2.553...)        
;                 ...
;        (1.939... 1.349... ~ 2.140... 2.073...)) :mgl t :shape (10 10))
@end[lang=lisp](code)
@end(section)"
  
  (call node (assure-tensor x)))

(defope !sin (SinTensor) node (x)
    "Applying sin to each element of x, creating a new sysconst.

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!randn `(5)))
;=>#Const((0.638... 0.527... 0.515... 0.495... 0.912...) :mgl t :shape (5))
(!sin a)
;=>#Const((-0.44... -0.64... -0.66... -0.70... -0.09...) :mgl t :shape (5))
@end[lang=lisp](code)
@end(section)"

  (call node (assure-tensor x)))

(defope !cos (CosTensor) node (x)
    "Applying cos to each element of x, creating a new sysconst.

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!randn `(5)))
;=>#Const((0.638... 0.527... 0.515... 0.495... 0.912...) :mgl t :shape (5))
(!cos a)
;=>#Const((0.803... 0.864... 0.870... 0.879... 0.611...) :mgl t :shape (5))
@end[lang=lisp](code)
@end(section)"

  (call node (assure-tensor x)))

(defope !tan (TanTensor) node (x)
    "Applying tan to each element of x, creating a new sysconst.

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!randn `(5)))
;=>#Const((0.638... 0.527... 0.515... 0.495... 0.912...) :mgl t :shape (5))
(!tan a)
;=>#Const((0.741... 0.582... 0.566... 0.540... 1.293...) :mgl t :shape (5))
@end[lang=lisp](code)
@end(section)"

  (call node (assure-tensor x)))

(defope !sinh (HyperbolicSinTensor) node (x)
    "Applying sinh to each element of x, creating a new sysconst.

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!randn `(5)))
;=>#Const((0.638... 0.527... 0.515... 0.495... 0.912...) :mgl t :shape (5))
(!sinh a)
;=>#Const((0.682... 0.551... 0.538... 0.516... 1.044...) :mgl t :shape (5))
@end[lang=lisp](code)
@end(section)"

  (call node (assure-tensor x)))

(defope !cosh (HyperbolicCosTensor) node (x)
    "Applying cosh to each element of x, creating a new sysconst.

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!randn `(5)))
;=>#Const((0.638... 0.527... 0.515... 0.495... 0.912...) :mgl t :shape (5))
(!cosh a)
;=>#Const((1.210... 1.142... 1.135... 1.125... 1.446...) :mgl t :shape (5))
@end[lang=lisp](code)
@end(section)"

  (call node (assure-tensor x)))

(defun !asin (x)
  "Applying asin to each element

asin(x) = 1/sin(x)"
  (!div 1 (!sin x)))

(defun !acos (x)
  "Applying acos to each element

acos(x) = 1/cos(x)"
  (!div 1 (!cos x)))

(defun !atan (x)
  "Applying atan to each element

atan(x) = 1/tan(x)"
  (!div 1 (!tan x)))

(defun !asinh (x)
  "Applying asinh to each element

asinh(x) = 1/sinh(x)"
  (!div 1 (!sinh x)))

(defun !acosh (x)
  "Applying acosh to each element

acosh(x) = 1/cosh(x)"
  (!div 1 (!cosh x)))

(defun !atanh (x)
  "Applying atanh to each element

atanh(x) = 1/tanh(x)"
  (!div 1 (!tanh x)))

(defun !argmaxmin (tensor max-or-min &key (dim nil))
  "Todo: For GPU"
  (declare (optimize (speed 3))
	   (type waffetensor tensor))
  (let* ((dim (if (and dim (< (the fixnum dim) 0))
		  (the fixnum (+ (the fixnum (!dims tensor))
				 (the fixnum dim)))
		  dim))
	 (result (!zeros (if (null dim)
			     '(1)
			     (or (loop for i fixnum upfrom 0 below (!dims tensor)
				       while (< i (the fixnum dim))
				       collect (!shape tensor i))
				 `(1)))))
	 (iter-num (/ (the fixnum (!size tensor))
		      (the fixnum (!size result))))
	 (dim (or dim 0)))

    (if (>= (the fixnum (or dim 0))
	    (the fixnum (!dims tensor)))
	(error "!argmax/min: the specified dim ~a is larger than ~a. Satisfy :dim (~a) < !dims (~a)"
	       dim
	       (!dims tensor)
	       dim
	       (!dims tensor)))
    (with-facet (return-array ((data result) 'array :direction :output))
      (labels ((apply-tensor (rest-dims apply-dims)
		 (loop for i fixnum upfrom 0 below (car rest-dims)
		       do (if (= (length rest-dims) 1)
					; the tensor now referring is the last.
			      (let* ((m-val nil)
				     (m-pos nil)
				     (ts (loop for i fixnum upfrom 0 below dim
					       collect t))
				     (result-dim `(,@apply-dims ,i))
				     (result-dim1 `(,@ts ,@(cdr result-dim)))
				     (result-dim1 (if (= dim 0)
						      result-dim1
						      result-dim)))
				(with-facet (arr
					     ((data (apply
						     #'!faref
						     tensor
						     result-dim1))
					      'backing-array
					      :direction :input))
				  (declare (type (simple-array single-float) arr))
				  (loop for m fixnum upfrom 0 below iter-num
					do (case max-or-min
					     (:max
					      (cond
						((null m-val)
						 (setq m-val (aref arr m))
						 (setq m-pos (+ 0.0 m)))
						((> (aref arr m) m-val)
						 (setq m-val (aref arr m))
						 (setq m-pos (+ 0.0 m)))))
					     (:min
					      (cond
						((null m-val)
						 (setq m-val (aref arr m))
						 (setq m-pos (+ 0.0 m)))
						((< (aref arr m) m-val)
						 (setq m-val (aref arr m))
						 (setq m-pos (+ 0.0 m)))))
					     (T (error "!argmaxmin, max-or-min is :max or :min."))))
				  
				  (apply
				   #'(setf aref)
				   m-pos
				   return-array
				   result-dim)))
					;else			      
			      (apply-tensor (cdr rest-dims)
					    `(,@apply-dims ,i))))))
	(apply-tensor (!shape result)
		      nil)
	result))))

(defun !argmax (tensor &key (dim nil) (keepdims nil))
  "Returns the indices of the maximum value of all elements in the input tensor.

@begin(deflist)
@def(dim)
@term(The dimension to reduce. If nil, the argmax of the flattened input is returned.)
@def(keepdims)
@term(whether the output tensor has dim retained or not. Ignored if dim=nil.)
@end(deflist)

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!randn `(5)))
;#Const((0.933... 0.158... 0.822... 0.881... 0.831...) :mgl t :shape (5))
(!argmax a)
;#Const((0.0) :mgl t :shape (1))
(setq a (!randn `(10 10 10)))
;#Const((((0.393... 0.658... ~ 0.003... 0.609...)         
;                   ...
;         (0.394... 0.252... ~ 0.688... 0.057...))        
;                 ...
;        ((0.325... 0.794... ~ 0.540... 0.381...)         
;                   ...
;         (0.310... 0.035... ~ 0.280... 0.431...))) :mgl t :shape (10 10 10))

(!argmax a :dim 2)

;#Const(((5.0 9.0 ~ 0.0 4.0)        
;                 ...
;        (2.0 0.0 ~ 2.0 5.0)) :mgl t :shape (10 10))

(!argmax a :dim 2 :keepdims t)
;#Const((((5.0 5.0 ~ 5.0 5.0)         
;                   ...
;         (4.0 4.0 ~ 4.0 4.0))        
;                 ...
;        ((2.0 2.0 ~ 2.0 2.0)         
;                   ...
;         (5.0 5.0 ~ 5.0 5.0))) :mgl t :shape (10 10 10))
@end[lang=lisp](code)
@end(section)"
  (if (null keepdims)
      (!argmaxmin tensor :max :dim dim)
      (!repeats (!unsqueeze (!argmaxmin tensor :max :dim dim) dim)
		dim
		(!shape tensor dim))))

(defun !argmin (tensor &key (dim nil) (keepdims nil))
  "Returns the indices of the minimum value of all elements in the input tensor.

@begin(deflist)
@def(dim)
@term(The dimension to reduce. If nil, the argmax of the flattened input is returned.)
@def(keepdims)
@term(whether the output tensor has dim retained or not. Ignored if dim=nil.)
@end(deflist)

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!randn `(5)))
;=>#Const((0.635... 0.101... 0.864... 0.563... 0.481...) :mgl t :shape (5))
(!argmin a)
;=>#Const((1.0) :mgl t :shape (1))

(setq a (!randn `(10 10 10)))
;#Const((((0.267... 0.113... ~ 0.142... 0.208...)         
;                   ...
;         (0.174... 0.948... ~ 0.232... 0.462...))        
;                 ...
;        ((0.454... 0.361... ~ 0.605... 0.731...)         
;                   ...
;         (0.099... 0.816... ~ 0.729... 0.996...))) :mgl t :shape (10 10 10))

(!argmin a)
;#Const((415.0...) :mgl t :shape (1))
@end[lang=lisp](code)
@end(section)"
  
  (if (null keepdims)
      (!argmaxmin tensor :min :dim dim)
      (!repeats (!unsqueeze (!argmaxmin tensor :min :dim dim) dim)
		dim
		(!shape tensor dim))))

(defope !abs (AbsTensor) node (x)
    "Computes the absolute value of each element in @cl:param(x).

Example:
@begin[lang=lisp](code)
(setq a (!random `(10 10) '(-1.0 1.0)))
;#Const(((0.048... 0.805... ~ 0.769... 0.252...)        
;                 ...
;        (0.159... -0.66... ~ -0.55... -0.23...)) :mgl t :shape (10 10))
(!abs a)
;#Const(((0.048... 0.805... ~ 0.769... 0.252...)        
;                 ...
;        (0.159... 0.667... ~ 0.553... 0.239...)) :mgl t :shape (10 10))
@end[lang=lisp](code)"
  (call node (assure-tensor x)))

(defun != () "Todo")
(defun !<= () "Todo")
(defun !>= () "Todo")


(defun get-sum-symbols (symbols)
  (let ((symbols (flatten symbols)))
    (map 'list
	 #'(lambda (x)
	     (setq symbols (delete x symbols :count 1)))
	 (remove-duplicates symbols))
    (remove-duplicates symbols)))

(defun execute-einsum-out (shapes-table
			   index-table
			   tensors
			   out)
  (print out)
  )

(defun buildup-einsum-node (shapes-table
			    index-table
			    operations
			    indices
			    explicts
			    outs
			    top-iters
			    tensors
			    result-table)
  (let ((result)
	(result-table (or result-table (make-hash-table)))
	(count 0))
    (mapc
     #'(lambda (index)
	 (typecase index
	   (list
	    (let* ((target-out1 (find index outs
				      :test #'(lambda (sym out)
						(eql sym (nth (1- (length out)) out)))))
		   (target-out (loop for i fixnum upfrom 0 below (length target-out1)
				     until (eql (nth i target-out1) '->)
				     collect (typecase (nth i target-out1)
					       (symbol (nth i target-out1))
					       (list (butlast (nth i target-out1))))))
		   (target-index (loop for i fixnum upfrom 0 below (length target-out1)
				       until (eql (nth i target-out1) '->)
				       collect (typecase (nth i target-out1)
						 (symbol (nth i target-out1))
						 (list (lastcar (nth i target-out1)))))))
	      (setf (gethash index index-table) count)
	      (buildup-einsum-node
	       shapes-table
	       index-table
	       target-out
	       target-index
	       (get-sum-symbols target-out)
	       outs
	       (loop for i fixnum upfrom (1+ (position '-> target-out1))
		     until (null (nth i target-out))
		     collect (nth i target-out1))
	       tensors
	       result-table))))
	 (incf count 1))
     indices)

					;(setf (gethash top-iters ~~))


    (print "evaluated")
					;    (print operations)
					;   (print indices)
					;(print explicts)
					;(print top-iters)

    
					;  (print (gethash (car explicts) index-table))

    (let ((target-tensor (map 'list #'(lambda (x)
					(print "INDEX")
					(print x)
					(typecase x
					  (fixnum (nth x tensors))
					  (list (gethash x result-table))))
			      indices)))
      (dolist (e explicts)
	(let ((num (gethash e index-table)))
	  (print num)))
      (print target-tensor)
      (print explicts)
      (setf (gethash explicts result-table)
	    (nth 0 tensors))))
  0)

(defmacro ->1 (einsum &rest args)
  "(-> (!einsum ~~) tensor1 tensor2)"
					; jitで高速化したい
					; IQ1e-10みたいな実装になった・・・
					; Todo: protect here with alexandria:once-only
  `(multiple-value-bind
	 (outs operations indices explicts top-iters subscripts)
       (funcall ,einsum (list ,@args))

     (let* ((shapes-table (make-hash-table))
	    (index-table  (make-hash-table)))
       (mapc
	#'(lambda (symbs tensor)
	    (loop for i fixnum upfrom 0 below (length symbs)
		  do (progn
		       (setf (gethash (nth i symbs) shapes-table)
			     (!shape tensor i)))))
	subscripts (list ,@args))

       (let ((top-shape
	       (map 'list #'(lambda (sym) (gethash sym shapes-table)) top-iters)))
	 (if (eql (caar explicts) 'nil)
					;sum mode
	     (let ((result))
	       (mapc
		#'(lambda (symb iter-num)
		    (loop for i fixnum upfrom 0 below iter-num
			  do (progn
			       (setf (gethash symb index-table) i)
			       (let ((output
				       (buildup-einsum-node
					shapes-table
					index-table
					operations
					indices
					explicts
					outs
					top-iters
					(list ,@args)
					nil)))
				 (if (null result)
				     (setq result output)
				     (setq result (!add result output)))))))
		top-iters top-shape)
	       result)
					;result is tensor
	     (let ((result))

	       ))))))

(defmacro -> (einsum &rest args)
  ""
  (declare (optimize (speed 3)))
  `(let ((einsum ,einsum)
	 ,@(map 'list #'(lambda (x) `(,x ,x)) args))
     (funcall einsum (list ,@args))))

(defmacro !einsum (&rest description)
  "Sums the product of the elements of the input operands along dimensions specified using a notation based on the Einstein summation convention.

See also: This operator is used with @cl:param(->).

方針: !arefと既存の命令でBackward可能にする。
!sumとか既存の命令に置き換えれる場合はそれに置き換える。
OutputのShapeは全て共通じゃないとダメ
... => ~ (repeat)"
  (unless (find '-> `,description)
					; Implict Mode
    (error "!einsum: Invaild Syntax. The required keyword -> is not found in given arguments ~a" `,description))

					; Explict Mode
					; parse arguments and check syntax.
  (let* ((subscripts (loop for i fixnum upfrom 0 below (length `,description)
			   until (equal '-> (nth i `,description))
			   collect (nth i `,description)))
	 (explicts   (loop for i fixnum upfrom (1+ (position '-> `,description))
			   until (null (nth i `,description))
			   collect (nth i `,description)))
	 (subscripts-indices
	   (loop for i fixnum upfrom 0 below (length subscripts)
		 collect i)))
    
    (unless explicts
      (push `(NIL) explicts))
    
    (map 'list #'(lambda (arg)
		   (typecase arg
		     (symbol nil)
		     (list
		      (map 'list #'(lambda (x)
				     (unless (or (typep x 'symbol)
						 (typep x 'list))
				       (error "!einsum: Invaild Syntax. !einsum excepts symbol or list as a subscripts but got ~a" x)))
			   arg)

		      (if (>= (count '~ arg) 2)
			  (error "!einsum: Invaild Syntax. ~a can be used at once in one subscriptions. at ~a" '~ arg))
		      nil)
		     (T
		      (error "!einsum: Invaild Syntax. !einsum excepts symbol or list as arguments but got ~a" `,arg))))
	 `,subscripts)
					; Optimize einsum Operations.

    (let* ((paths)) ; paths -> (tensors) -> (values n paths)
      (labels ((create-path-lambda (o-exp outs opes indices exps
				    &aux (routs (map 'list
						     #'(lambda (x)
							 (cddr (reverse x)))
						     outs)))
		 #'(lambda (tensors)
		     (let ((ctime 0))
		       (dolist (o o-exp)
			 (when (eql (car o) :*)
			   (incf ctime (apply #'*
					      (let ((index 0))
						(mapc
						 #'(lambda (o1)
						     (mapc
						      #'(lambda (t1)
							  (if (eql (car t1) (cdr o))
							      (setq index (second t1))))
						      o1))
						 routs)
						(!shape (nth index tensors)))))))
		       (values ctime
			       (list
				outs
				opes
				indices
				exps
				(get-sum-symbols (cddr (reverse description))))))))
	       (get-sum-symbols (symbols)
		 (let ((symbols (flatten symbols)))
		   (map 'list
			#'(lambda (x)
			    (setq symbols (delete x symbols :count 1)))
			(remove-duplicates symbols))
		   (remove-duplicates symbols)))
	       (butnth (list n &optional (n1 nil))
		 (loop for i fixnum upfrom 0 below (length list)
		       when (not (or (= i n)
				     (= i (or n1 -1))))
			 collect (nth i list)))
	       (afternth (list n)
		 (loop for i fixnum upfrom 0 below (length list)
		       when (< i n)
			 collect (nth i list)))
	       (get-computation-time (out)
		 (let* ((iter-symbols (nth (1- (length out)) out)))
		   `(:* ,@iter-symbols)))
	       (explore-ctime (explicts indices outs)
		 (map 'list
		      #'(lambda (index)
			  (typecase index
			    (fixnum (get-computation-time explicts))
			    (list (get-computation-time
				   (find index outs
					 :test #'(lambda (s o)
						   (equal
						    s
						    (nth (1- (length o)) o))))))))
		      indices))
	       (explore-path (operations indices outs)
		 (if (> (length operations) 2)
					; subscriptions are larger than 2.
		     (dotimes (i (length operations))
		       (let ((rest (afternth operations i))
			     (rest-i (afternth indices i))
			     (ith-operation (nth i operations))
			     (ope-i (nth i indices)))
			 (dotimes (k (length rest))
			   (let ((pair (nth k rest))
				 (pair-i (nth k rest-i))
				 (others (butnth operations i k))
				 (others-i (butnth indices i k)))

			     (let* ((next-out (get-sum-symbols `(,@ith-operation ,@pair)))
				    (others-next `(,next-out ,@others)))
			       (unless (null next-out) ; maybe reducible
				 (explore-path
				  others-next
				  `(,next-out ,@others-i)
				  `(,@outs ; 上から下に実行
				    ((,ith-operation ,ope-i)
				     (,pair ,pair-i)
				     ->
				     ,next-out)))))))))
		     (let ((outs (or outs operations)))		       
					; reached AB -> C 

		       (print `(,@operations -> ,@explicts))
		       (print `(,@indices -> ,@explicts)) ; result

		       (print indices)
		       (print outs)
		       (print (explore-ctime explicts indices outs))
		       (push (create-path-lambda
			      (explore-ctime explicts indices outs)
			      outs
			      operations
			      indices
			      explicts)
			     paths)))))
	(explore-path `,subscripts
		      subscripts-indices
		      nil)
	#'(lambda (tensors)
	    (unless (= (length tensors)
		       (length `,subscripts))
	      (error "!einsum: The size of subscripts and tensor doesn't match."))

	    (multiple-value-bind (p code) (funcall (car paths) tensors)
	      (let ((path p)
		    (code code))
		(dolist (p (cdr paths))
		  (multiple-value-bind (n1 code1) (funcall p tensors)
		    (if (< n1 path)
			(progn
			  (setq path n1)
			  (setq code code1)))))
		(let ((outs (nth 0 code))
		      (top-operation (nth 1 code))
		      (top-operation-indices (nth 2 code))
		      (top-explicts (nth 3 code))
		      (top-iters (nth 4 code)))
		  (values outs
			  top-operation
			  top-operation-indices
			  top-explicts
			  top-iters
			  `,subscripts)))))))))

(defmacro !einsum1 (&rest description)
  ""
  (declare (optimize (speed 3))
	   (type list description))
  (let* ((subscripts (loop for i fixnum upfrom 0 below (length `,description)
			   until (equal '-> (nth i `,description))
			   collect (nth i `,description)))
	 (explicts   (loop for i fixnum upfrom (1+ (position '-> `,description))
			   until (null (nth i `,description))
			   collect (nth i `,description)))
	 (iter-symbols (get-sum-symbols subscripts)))
    (declare (type list subscripts iter-symbols))
    (labels ((get-subscript-index (tensors symbol)
	       (declare (type list tensors)
			(type symbol symbol))
	       (loop named sloop
		     for i fixnum upfrom 0 below (length subscripts)
		     do (loop with ith-tensor = (nth i tensors)
			      for m fixnum
			      upfrom 0
				below (length (the list (nth i subscripts)))
			      do (let ((mth-symbol (nth m (nth i subscripts))))
				   (if (eql symbol mth-symbol)
				       (let ((size (!shape ith-tensor m)))
					 (declare (type fixnum size))
					 (return-from
					  sloop size)))))))
	     (get-subscript-index-iter (tensors symbol nth)
	       (declare (type symbol symbol))
	       (if (find symbol iter-symbols)
		   1
		   (or (get-subscript-index tensors symbol)
		       (shape-nth tensors nth))))
	     (shape-nth (tensors n)
	       (declare (type fixnum n))
	       (loop for i fixnum upfrom 0 below n
		     maximize (let ((res (!shape (nth i tensors) n)))
				(declare (type fixnum res))
				res)))
	     (parse-subscripts (n)
	       (nth n subscripts))
	     (parse-explicts (indices)
	       (map 'list #'(lambda (x)
			      (declare (type symbol x))
			      (if (find x iter-symbols)
					; Sum up about x
				  (nth (position x (the list (car explicts))) indices)
				  t))
		    (car explicts))))

      #'(lambda (tensors)
	  (declare (optimize (speed 3))
		   (type list tensors))
	  (let* ((result-dim (loop for m fixnum
				   upfrom 0
				     below (length
					    (the list (car explicts)))
				   collect (get-subscript-index-iter tensors (nth m (car explicts)) m)))
		 (result (!zeros result-dim)))
	    (labels ((sumup-next-iter (symbols &optional (indices nil))
		       (declare (optimize (speed 3)))
		       (loop with symbol = (car symbols)
			     for i fixnum
			     upfrom 0
			       below (get-subscript-index tensors symbol)
			     unless (null (cddr symbols)) ; remains > 2d
			       do  (sumup-next-iter
				    (cdr symbols)
				    `(,@indices ,i))
			     else
			       do (loop with indices = `(,@indices ,i)
					with tmp = nil
					for nth fixnum upfrom 0 below (length tensors)
					do (let* ((args-sub (parse-subscripts nth))
						  (exps-sub (parse-explicts args-sub))
						  (sumup-mode (= (the fixnum (apply #'* result-dim)) 1))
						  (value (apply
							  #'!aref
							  (nth nth tensors)
							  indices))
						  (init-it (= nth 0))
						  (transpose-point (loop for s fixnum upfrom 0 below (length (the list args-sub))
									 minimize (if (eql (the symbol (nth s args-sub)) (the symbol (nth s (car explicts))))
										      (1+ (length (the list args-sub)))
										      s)))
						  (transpose-point (if (= transpose-point (1+ (length (the list args-sub))))
								       nil
								       transpose-point)))
					    ; (print transpose-point)
					     ;(print args-sub)
					     ;(print exps-sub)
					     ;(print (car explicts))
					     ;(print tmp)
					     ;(print value)

					     (unless (null transpose-point)
					       (let ((shape (copy-list exps-sub)))
						 (setf (nth transpose-point shape) (!size value))
						 (setq value
						       (!reshape value shape))))

					     (setf (nth (case transpose-point
							  (0 1)
							  (1 0)
							  (T 0))
							exps-sub)
						   i)
					     (if init-it
						 (setq tmp value)
						 (setq tmp (!mul tmp value)))

					     (when (= nth (1- (length tensors))) ;reached an last term
					       (if sumup-mode
						   (setq result (!sum tmp))
						   (apply
						    #'(setf !aref)
						    tmp
						    result
						    exps-sub)
						  )))))))
	      (sumup-next-iter
	       (or iter-symbols
		   (car subscripts)))
	      result))))))

(defun !ravel () "Todo")
(defun !flatten (tensor)
  "Flattens input by reshaping it into a one-dimensional tensor.

The operation is the same as @c((!reshape tensor '(t)))

Example:
@begin[lang=lisp](code)

(setq a (!randn `(10 10)))
;#Const(((0.688... 0.580... ~ 0.013... 0.461...)        
;                 ...
;        (0.214... 0.248... ~ 0.540... 0.416...)) :mgl t :shape (10 10))

(!flatten a)
;#Const((0.688... 0.580... ~ 0.540... 0.416...) :mgl t :shape (100))
@end[lang=lisp](code)"
  (!reshape tensor '(t)))

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

