
(in-package :cl-waffe)

#|
Here's elementary operators and nodes for tensor.
And utils for broadcasting etc...
|#

(declaim (ftype (function (t) waffetensor) assure-tensor))
(defun assure-tensor (x)
  "This function is used in order to implement this: e.g. (!add 1 1)"
  (typecase x
    (waffetensor x)
    (T (const x))))

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

(defun broadcasting (x y)
  (declare (type waffetensor x y))
  (map 'list #'(lambda (xi yi)
		 (declare (type fixnum xi yi))
		 (if (and (or (= xi 1) (= yi 1))
			  (not (= xi yi)))
		     (if (= xi 1)
			 `(,(max xi yi) nil)
			 `(nil ,(max xi yi)))
		     (if (= xi yi)
			 `(nil nil)
			 (error "cl-waffe.broadcasting: can't broadcast two tensors ~a and ~a." x y))))
       (!shape x) (!shape y)))

(defun sumup-broadcasted (list x y)
  (let ((x x)
	(y y))
    (loop for dim fixnum upfrom 0 below (!dims x)
	  do (let ((b (nth dim list)))
	       (unless (null (car b))
		 (setq x (!mean x dim)))
	       (unless (null (second b))
		 (setq y (!mean y dim)))))
  (list x y)))

(defun straighten-up (x y)
  "Straigthen up the dims of x and y"
  (let ((dims-x (!dims x))
	(dims-y (!dims y)))
    (cond
      ((= dims-x dims-y)
       (values x y))
      ((> dims-x dims-y)
       (let ((count (- dims-x dims-y)))
	 (!allow-destruct y)
	 (values x (prog1
		       (!unsqueeze y 0 count)
		     (!disallow-destruct y)))))
      ((> dims-y dims-x)
       (let ((count (- dims-y dims-x)))
	 (!allow-destruct x)
	 (values (prog1
		     (!unsqueeze x 0 count)
		   (!disallow-destruct x))
		 y))))))

(defun same-shape-p (x y)
  (declare (optimize (speed 3))
	   (type waffetensor x y))
  (or
   (or (not (typep (data x) 'mat))
       (not (typep (data y) 'mat)))
   (equal (the list (!shape x)) (the list (!shape y)))))

(defmacro k-> (kernel-function output overwrite &rest args)
  "Alias for call-and-dispatch-kernel"
  `(call-and-dispatch-kernel
    ,kernel-function ,output ,overwrite ,@args))

(defun sumup-tensor (x)
  (declare (type waffetensor x))
  (let ((r 0.0))
    (declare (type single-float r))
    (with-facet (arr ((value x) 'backing-array :direction :input))
      (let ((size (array-total-size arr)))
	(loop for i fixnum upfrom 0 below size
	      do (incf r (aref arr i)))))
    (sysconst r)))

(defnode AddTensor (&optional (output nil) (overwrite nil))
  :optimize t
  :parameters ((output output) (overwrite overwrite))
  :forward  ((x y)
	     (k-> :add (self output) (self overwrite) x y))
  :backward ((dy) (list dy dy)))

(defnode SubTensor (&optional (output nil) (overwrite nil))
  :optimize t
  :parameters ((output output) (overwrite overwrite))
  :forward ((x y)
	    (k-> :sub (self output) (self overwrite) x y))
  :backward ((dy) (list dy (!mul dy (const -1)))))

(defnode MulTensor (&optional (output nil) (overwrite nil))
  :optimize t
  :parameters ((xi T) (yi T) (output output) (overwrite overwrite))
  :forward ((x y)
	    (save-for-backward xi x)
	    (save-for-backward yi y)
	    (k-> :mul (self output) (self overwrite) x y))
  :backward ((dy) (list (!mul (self yi) dy)
			(!mul (self xi) dy))))

(defnode BroadCastingAddTensor (&optional (output nil) (overwrite nil))
  :optimize t
  :parameters ((dims-to-sum nil) (output output) (overwrite overwrite))
  :forward  ((x y)
	     (let ((dims-to-sum (broadcasting x y)))
	       (setf (self dims-to-sum) dims-to-sum)
	       (k-> :add (self output) (self overwrite) x y)))
  :backward ((dy) (sumup-broadcasted (self dims-to-sum) dy dy)))

(defnode BroadCastingSubTensor (&optional (output nil) (overwrite nil))
  :optimize t
  :parameters ((dims-to-sum nil) (output output) (overwrite overwrite))
  :forward ((x y)
	    (let ((dims-to-sum (broadcasting x y)))
	      (setf (self dims-to-sum) dims-to-sum)
	      (k-> :sub (self output) (self overwrite) x y)))
  :backward ((dy) (sumup-broadcasted (self dims-to-sum) dy (!mul dy (const -1)))))

(defnode BroadCastingMulTensor (&optional (output nil) (overwrite nil))
  :optimize t
  :parameters ((xi T) (yi T) (dims-to-sum nil) (output output) (overwrite overwrite))
  :forward ((x y)
	    (let ((dims-to-sum (broadcasting x y)))
	      (setf (self dims-to-sum) dims-to-sum)
	      (save-for-backward xi x)
	      (save-for-backward yi y)
	      (k-> :mul (self output) (self overwrite) x y)))
  :backward ((dy) (sumup-broadcasted (self dims-to-sum)
				     (!mul (self yi) dy)
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

(defnode TransposeOriginalTensor (shape)
  :optimize t
  :parameters ((prev-shape nil) (shape shape))
  :forward ((x)
	    (setf (self prev-shape) (!shape x))
	    (with-facet (array ((value x) 'array :direction :input))
	      (sysconst (array-to-mat (numcl:transpose array)))))
  :backward ((dy)
	     (list (!transpose1 dy (self prev-shape)))))

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
		(sumup-tensor x))
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

(defnode ASinTensor ()
  :parameters ((xi nil))
  :forward ((x)
	    (save-for-backward xi x)
	    (with-searching-calc-node :asin x))
  :backward ((dy)
	     (list (!mul dy (!acos (self x))))))

(defnode ACosTensor ()
  :parameters ((xi nil))
  :forward ((x)
	    (save-for-backward xi x)
	    (with-searching-calc-node :acos x))
  :backward ((dy)
	     (list (!mul dy (!mul -1.0 (!asin (self x)))))))

(defnode ATanTensor ()
  :parameters ((xi nil))
  :forward ((x)
	    (save-for-backward xi x)
	    (with-searching-calc-node :atan x))
  :backward ((dy)
	     (list (!mul dy (!div 1 (!pow (!acos (self x)) 2))))))

(defnode ASinhTensor ()
  :parameters ((xi nil))
  :forward ((x)
	    (save-for-backward xi x)
	    (with-searching-calc-node :asinh x))
  :backward ((dy)
	     (list (!mul dy (!acosh (self x))))))

(defnode ACoshTensor ()
  :parameters ((xi nil))
  :forward ((x)
	    (save-for-backward xi x)
	    (with-searching-calc-node :acosh x))
  :backward ((dy)
	     (list (!mul dy (!mul -1.0 (!asinh (self x)))))))

(defnode ATanhTensor ()
  :parameters ((xi nil))
  :forward ((x)
	    (save-for-backward xi x)
	    (with-searching-calc-node :atanh x))
  :backward ((dy)
	     (list (!mul dy (!div 1 (!pow (!acosh (self x)) 2))))))

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
       (declare (optimize (speed 3) (safety 1)))
       (let* ((,tensor (if *no-grad* ,place ,node-object)))
	 ,@body))))

(defope !add (AddTensor) node (x y)
    "Adds x and y.

In the case when x or y is not a tensor, automatically creates a new tensor.

Destructive mode: (!!add x y)

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
  (let ((x (assure-tensor x))
	(y (assure-tensor y)))
    (if (same-shape-p x y)
	(call node x y)
	(multiple-value-bind (x y) (straighten-up x y)
	  (call (BroadCastingAddTensor) x y)))))

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
  (let ((x (assure-tensor x))
	(y (assure-tensor y)))
    (if (same-shape-p x y)
	(call node x y)
	(multiple-value-bind (x y) (straighten-up x y)
	  (call (BroadCastingSubTensor) x y)))))

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
  (let ((x (assure-tensor x))
	(y (assure-tensor y)))
    (if (same-shape-p x y)
	(call node x y)
	(multiple-value-bind (x y) (straighten-up x y)
	  (call (BroadCastingMulTensor) x y)))))

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
  (let ((result (!div-old 1 y)))
    (!allow-destruct result)
    (!mul result x)))

(defun !!div (target-x target-y)
    "Divides target-x by target-y in a destructive way.

target-x and target-y are always substituted for the result

See also: @link[uri=\"./using-tensor.html#compute-tensors-in-a-destructive-way\"](Destructive Operations)"
  (let ((target-x (assure-tensor target-x))
	(target-y (assure-tensor target-y)))
    (!allow-destruct target-x)
    (!allow-destruct target-y)
    (!!mul target-x (!div-old 1 target-y))))

(defun !!inv () "Todo")

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

(defun !!add (target-x y)
  "Adds target-x and y in a destructive way.

target-x is always substituted for the result

y is not subject to side effects unless target-x is not a mat.

See also: @link[uri=\"./using-tensor.html#compute-tensors-in-a-destructive-way\"](Destructive Operations)"
  (let ((x (assure-tensor target-x))
	(y (assure-tensor y)))
    (if (same-shape-p x y)
	(call (AddTensor x t) x y)
	(multiple-value-bind (x y) (straighten-up x y)
	  (setf (waffetensor-is-next-destruct? x) t)
	  (let ((result (call (BroadCastingAddTensor x t) x y)))
	    (setf (data x) result)
	    (the waffetensor result))))))

(defun !!sub (target-x y)
  "Substracts target-x by y in a destructive way.

target-x is always substituted for the result.

y is not subject to side effects unless target-x is not a mat.

See also: @link[uri=\"./using-tensor.html#compute-tensors-in-a-destructive-way\"](Destructive Operations)"
  (let ((x (assure-tensor target-x))
	(y (assure-tensor y)))
    (if (same-shape-p x y)
	(call (SubTensor x t) x y)
	(multiple-value-bind (x y) (straighten-up x y)
	  (setf (waffetensor-is-next-destruct? x) t)
	  (let ((result (call (BroadCastingSubTensor x t) x y)))
	    (setf (data x) result)
	    (the waffetensor result))))))

(defun !!mul (target-x y)
    "Multiplys target-x and y in a destructive way.

target-x is always substituted for the result

y is not subject to side effects unless target-x is not a mat.

See also: @link[uri=\"./using-tensor.html#compute-tensors-in-a-destructive-way\"](Destructive Operations)"
  (let ((x (assure-tensor target-x))
	(y (assure-tensor y)))
    (if (same-shape-p x y)
	(call (MulTensor x t) x y)
	(multiple-value-bind (x y) (straighten-up x y)
	  (setf (waffetensor-is-next-destruct? x) t)
	  (let ((result (call (BroadCastingMulTensor x t) x y)))
	    (setf (data x) result)
	    (the waffetensor result))))))

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
    (1 (!sum-2d x axis keepdims))
    (2 (!sum-2d x axis keepdims))
    (T
     (if (null axis)
	 (call (SumUpTensor) x)
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

(defun !!pow (target-x n)
  "Takes the power of each element in @cl:param(x) with n.

target-x is destructed."

  (let ((target-x (assure-tensor target-x)))
    (!allow-destruct target-x)
    (!pow target-x n)))

(defope !sqrt (SqrtTensor) node (x)
    "Takes the power of each element in @cl:param(x) with 1/2, creating new sysconst and nodes.

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

(defun !!sqrt (target-x)
  "Takes the power of each element in @cl:param(x) with 1/2.

target-x is destructed."
  (let ((target-x (assure-tensor target-x)))
    (!allow-destruct target-x)
    (!sqrt target-x)))

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

(defun !!log (target-x)
  "Returns a modified tenssor with the natural logarithm of the elements of target-x"
  (let ((target-x (assure-tensor target-x)))
    (!allow-destruct target-x)
    (!log target-x)))

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

x can be: mat or tensor.

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

(!repeats (const 10.0) 3 10)
;#Const(((((10.0 10.0 ~ 10.0 10.0)))) :mgl t :shape (1 1 1 10))
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

(defun !transpose1 (x &rest result)
  "Transpose x but doesn't produce lazy-eval.

Todo: Numcl's operation couldm't optimized well. i need to reimplement it by myself.

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!randn `(10 5 3)))

(!transpose1 a)
;#Const((((-0.47... -0.03... ~ -0.17... 0.328...)         
;                   ...
;         (0.210... -1.80... ~ 1.648... 0.135...))        
;                 ...
;        ((-0.52... 1.509... ~ 0.643... 0.258...)         
;                   ...
;         (-0.26... -1.14... ~ -1.08... 1.126...))) :mgl t :shape (3 5 10))
@end[lang=lisp](code)
@end(section)"
  (call (TransposeOriginalTensor (assure-tensor result)) (assure-tensor x)))

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

@def(x is 3D and y is 3D.)
@begin(term)
The Batch Filtered Matrix-Matrix product is returned.
@begin[lang=lisp](code)
(setq a (!randn `(5 3 10)))
(setq b (!randn `(5 10 3)))

; The returned mat is comprised of:
; (!matmul (!aref a 0) (!aref b 0))
; (!matmul (!aref a 1) (!aref b 1))
; (!matmul (!aref a 2) (!aref b 2))
; (!matmul (!aref a 3) (!aref b 3))

(!matmul a b)
;#Const((((6.621... -5.61... 2.898...)         
;                   ...
;         (-2.96... -4.26... -3.99...))        
;                 ...
;        ((-0.02... 2.707... 5.989...)         
;                   ...
;         (-3.35... 3.561... -3.90...))) :mgl t :shape (5 3 3))
@end[lang=lisp](code)
@end(term)

@def(Otherwise)
@term(Currently not implemented. In the near future for more will be added.)
@end(deflist)"
  (cond
    ((and (= (the fixnum (!dims x)) 1)
	  (= (the fixnum (!dims y)) 1))
     (!dot x y))
    (T (call node (assure-tensor x) (assure-tensor y)))))

(defun !unsqueeze (x &optional (dim 0) (count 1))
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
    (dotimes (_ count)
      (case dim
	(0  (setq s `(1 ,@s)))
	(-1 (push 1 (cdr (nthcdr (1- (length s)) s))))
	(T  (push 1 (cdr (nthcdr (1- dim) s))))))
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

(defun !!exp (target-x)
  "Applying !exp in a destructive way."
  (let ((target-x (assure-tensor target-x)))
    (!allow-destruct target-x)
    (call (ExpTensor) target-x)))

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
  "Applying asin to each element"
  (call (ASinTensor) (assure-tensor x)))

(defun !acos (x)
  "Applying acos to each element"
  (call (ACosTensor) (assure-tensor x)))

(defun !atan (x)
  "Applying atan to each element"
  (call (ATanTensor) (assure-tensor x)))

(defun !asinh (x)
  "Applying asinh to each element"
  (call (ASinhTensor) (assure-tensor x)))

(defun !acosh (x)
  "Applying acosh to each element"
  (call (ACoshTensor) (assure-tensor x)))

(defun !atanh (x)
  "Applying atanh to each element"
  (call (ATanhTensor) (assure-tensor x)))


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

(defun range-tail (start end acc step)
  (if (> start end)
      acc
      (range-tail start (- end step) (cons end acc) step)))

(defun range (start end &optional (step 1) (fill nil))
  (range-tail start end '() step))
#|
(defnode FilteringTensorBackward (dim iter-num args)
  :parameters ((dim dim :type fixnum)
	       (iter-num iter-num :type fixnum)
	       (args args :type cons))
  :forward ((&rest result)
	    result)
  :backward ((dy)
	     (loop for i fixnum upfrom 0 below (self iter-num)
		   collect (apply #'!aref `(,(self args) ,i)))))

(defun !filter-tensor (tensor dim batch-size function)
  "This is a intrinsical function of doing iteration for waffe-tensor.
This is the very fastst but not useful. So use macros in order to make it more useful."
  (let ((args (loop for i fixnum upfrom 0 below (max (1- dim) 0)
		    collect t)))
    (print args)
    (apply #'call `(,(FilteringTensorBackward
		      dim
		      ,(/ (!shape tensor dim) batch-size)))
	   (loop for i fixnum
		 upfrom 0
		   below (/ (!shape tensor dim) batch-size)
		 collect
		 (funcall function
			  i
			  (apply #'!aref tensor `(,@args ,i))))))
|#
(defun !dotensor () "")
(defun !displace ()
  "")

(defun get-sum-symbols (symbols)
  (let ((symbols (flatten symbols)))
    (map 'list
	 #'(lambda (x)
	     (setq symbols (delete x symbols :count 1)))
	 (remove-duplicates symbols))
    (remove-duplicates symbols)))

(defmacro -> (einsum &rest args)
  "do not use this."
  (declare (optimize (speed 3)))
  `(let ((einsum ,einsum)
	 ,@(map 'list #'(lambda (x) `(,x ,x)) args))
     (funcall einsum (list ,@args))))

(defmacro !einsum (&rest description)
  "do not use this."
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

