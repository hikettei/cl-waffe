
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

; symbols for !modify
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

(defun broadcasting (x y)
  "Returns the list which indicates dims to be broadcasted"
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
  "Given list which is the return of broadcasting, sum up x and y (for backward !add/!sub...)"
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

(defnode ConcatenateTensorNode (axis)
  :optimize t
  :parameters ((axis axis :type fixnum)
	       (shape t))
  :forward ((&rest tensors)
	    (let ((first-shape (the list (!shape (car tensors)))))
	      (loop for i fixnum upfrom 0 below (length tensors)
		    unless (equal first-shape (the list (!shape (nth i tensors))))
		      do (error "cl-waffe.stack!: all tensors must be consisted of the same shapes, but got ~a. excepted:~a" (nth i tensors) first-shape))
	      (let* ((result-shape
		       (loop for i fixnum
			     upfrom 0
			       below (length first-shape)
			     if (= i (self axis))
			       collect (the fixnum (* (the fixnum (length tensors)) (the fixnum (nth i first-shape))))
			     else
			       collect (nth i first-shape)))
		     (result (!zeros result-shape)))
		(setf (self shape) (!shape (car tensors) (self axis)))
		(stack! (self axis)
			(map 'list #'value tensors)
			(data result))
		result)))
  :backward ((dy)
	     (!split dy (self shape) :axis (self axis))))

(defnode SplitTensorNode (split-size axis)
  :parameters ((split-size split-size :type fixnum)
	       (axis axis :type fixnum)
	       (prev-shape t))
  :forward ((tensor)
	    (setf (self prev-shape) (!shape tensor))
	    (loop
	      with each-tensor-shape = (let ((shape (copy-list (!shape tensor))))
					 (setf (nth (self axis) shape) (self split-size))
					 shape)
	      with tmp-areas = (loop for i fixnum upfrom 0 below (self axis)
				     collect t)
	      for ith fixnum
	      upfrom 0
		below (the fixnum
			   (multiple-value-bind (k rest)
			       (round
				(/
				 (!shape tensor (self axis))
				 (self split-size)))
			     
			     (declare (type fixnum k))
			     (if (> rest 0.0)
				 (1+ k)
				 k)))
	      collect (let ((start-index (* ith (self split-size)))
			    (end-index (* (1+ ith) (self split-size))))

			(if (<= end-index (!shape tensor (self axis)))
			    (apply
			     #'%saref
			     nil
			     tensor
			     `(,@tmp-areas (,start-index
					    ,end-index)))
			    (let ((res (!zeros each-tensor-shape)))
			      (copy!
			       (data
				(apply
				 #'%saref
				 nil
				 tensor
				 `(,@tmp-areas
				   (,start-index
				    ,(!shape tensor (self axis))))))
			       (data res))
			      res)))))
  :backward ((dy)
	     ; 枝分かれしてるTensorはbackward1関数内でしかaxpyされないはずだから、メモリ節約のためにdisplaceできない？
	     (error "SplitTensorNodeBackward is not yet implemented.~%Alternatively, !aref, (setf !aref) is available for a while.")
	     (list dy)))

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

(defun !concatenate (axis &rest tensors)
  "Concatenates the given sequence of @cl:param(tensors) in the given @cl:param(axis). All tensors must have the same shape.

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!randn `(3 3 3)))
;#Const((((1.000... -0.00... -0.25...)         
;                   ...
;         (1.473... -0.44... 1.680...))        
;                 ...
;        ((0.569... 0.852... 0.405...)         
;                   ...
;         (0.024... 0.756... 0.383...))) :mgl t :shape (3 3 3))

(!concatenate 0 a a a)
;#Const((((1.000... -0.00... -0.25...)         
;                   ...
;         (1.473... -0.44... 1.680...))        
;                 ...
;        ((0.569... 0.852... 0.405...)         
;                   ...
;         (0.024... 0.756... 0.383...))) :mgl t :shape (9 3 3))

(mgl-mat:M= (data (!aref * '(0 3)))
            (data (!aref * '(3 6))))
;T
@end[lang=lisp](code)
@end(section)"
  (declare (optimize (speed 3))
	   (type fixnum axis))
  (apply #'call (ConcatenateTensorNode axis) tensors))

(defun !stack (axis &rest tensors)
  "Stacks the given @cl:param(tensors) in the specified @cl:param(axis).

Internally, !stack @b(adds 1 to the specified axis) before calling !concatenate.

Note: Currently, when unsqueezing given tensors, !stack creates copies every time in order to prevent side effects. To avoid this, !concatenate is recommended to use. @b((TO FIX))

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!randn `(2 2 2)))

;#Const((((-0.83... -1.74...)
;         (0.119... 0.162...))
;        ((-1.81... 0.907...)
;         (-0.50... -0.96...))) :mgl t :shape (2 2 2))

(!stack 0 a a a)
;#Const(((((-0.83... -1.74...)
;          (0.119... 0.162...))
;         ((-1.81... 0.907...)
;          (-0.50... -0.96...)))        
;                 ...
;        (((-0.83... -1.74...)
;          (0.119... 0.162...))
;         ((-1.81... 0.907...)
;          (-0.50... -0.96...)))) :mgl t :shape (3 2 2 2))

(mgl-mat:M= (data (!aref * 0)) (data (!aref * 1)))
; T
@end[lang=lisp](code)
@end(section)"
  (let ((tensors (map 'list
		      #'(lambda (tensor)
			  ; bug: (!stack 0 a a a) <- reshaped for multiple times.
			  (!disallow-destruct tensor)
			  (!unsqueeze tensor axis))
		      tensors)))
    (apply #'call (ConcatenateTensorNode axis) tensors)))

(defun !split (tensor split-size &key (axis 0))
  "Splits the tensor into chunks in the specified @cl:param(axis). Each chunk is a copy of original tensor.

split-size indicates the strides of each chunk, that is, @cl:param(tensor) will be split into equalliy size of @cl:param(split-size).

split-size must be fixnum.

Note: currently !split's backward returns error. this is because there's a room to optimize backward and until optimized them i wont make it.

Alternatively, !aref, (setf !aref) is available.

@begin(section)
@title(Example)
@begin[lang=lisp](code)

(setq a (!randn `(4 2 2)))
;#Const((((-0.48... -1.22...)
;         (0.251... 0.476...))        
;                 ...
;        ((-0.66... 1.045...)
;         (-0.44... 1.592...))) :mgl t :shape (4 2 2))

(!split a 2)
;(#Const((((-0.48... -1.22...)
;         (0.251... 0.476...))
;        ((0.864... -0.93...)
;         (-0.43... 0.346...))) :mgl t :shape (2 2 2))
; #Const((((-1.91... -0.63...)
;         (-0.08... 0.867...))
;        ((-0.66... 1.045...)
;         (-0.44... 1.592...))) :mgl t :shape (2 2 2)))

; the rests are filled with 0.0
(!split a 3)
;(#Const((((-0.48... -1.22...)
;         (0.251... 0.476...))        
;                 ...
;        ((-1.91... -0.63...)
;         (-0.08... 0.867...))) :mgl t :shape (3 2 2))
; #Const((((-0.66... 1.045...)
;         (-0.44... 1.592...))        
;                 ...
;        ((0.0 0.0)
;         (0.0 0.0))) :mgl t :shape (3 2 2)))
@end[lang=lisp](code)
@end(section)"
  (declare (type waffetensor tensor)
	   (type fixnum split-size axis)
	   (optimize (speed 3)))
  (call (SplitTensorNode split-size axis) tensor))

(defmacro !vstack (&rest tensors)
  "!vstack is the equivalent to !concatenate(axis=0)"
  `(!concatenate 0 ,@tensors))

(defmacro !hstack (&rest tensors)
  "!vstack is the equivalent to !concatenate(axis=1)"
  `(!concatenate 1 ,@tensors))

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
(defun !displace () "")

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

