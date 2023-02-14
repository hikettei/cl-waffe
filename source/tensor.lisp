
(in-package :cl-waffe)


(defparameter *print-char-max-len* 5
  "When printing tensor, the character displayed following this param.
(e.g. When 5, in your terminal, 1.12345d0 => 1.1234...)
Default: 5")

(defparameter *print-arr-max-size* 6
  "When printing tensor, the tensor displayed following this param.
(e.g. When 5, in your terminal, (1 2 3 4 5 6 7 8 9 10) => (1 2 3 ... 4 5 6))
Default: 6")

(defparameter *print-mat-max-size* 3
  "When printing tensor, the tensor displayed following this param.
(e.g. When 3, in your terminal, ((1) (2) (3) (4)) => ((1) (2) ... (4)))")

(defparameter *default-backend* :mgl
  "Default backend cl-waffe uses. Default: :mgl")

(defparameter mgl-mat:*DEFAULT-MAT-CTYPE* :float) ; in default, float

(deftype WaffeSupportedDataType ()
  "An type of waffe-tensor's content type,

`(or fixnum float null cons function ratio)"
  `(or fixnum float null cons function ratio))

(deftype WaffeDataType ()
  `(or fixnum
       float
       null
       cons
       function
       ratio))

(deftype waffe-array () ;  an list of waffe supported array data structures.
  `(or mgl-mat:mat))

(defun waffe-array (c)
  (and (typep c 'simple-array)
       (every (lambda (e) (typep e 'waffesupporteddatatype)) c)))

(deftype WaffeTensorContentType ()
  "An type of data that allowed to make tensors with (const ~) or (tensor ~).

cl-waffe automatically coerce them to arbitary types

`(or mgl-mat:mat
     simple-array
     waffesupporteddatatype)"
  `(or mgl-mat:mat
       simple-array
       waffesupporteddatatype))

(deftype WaffeTensorTypes ()
  `(or mgl-mat:mat waffesupporteddatatype))

(defstruct grad-tmp
  (value nil)
  (grad-called nil :type boolean))

(defstruct (WaffeNodeThread
	    (:constructor
		thread
		(thread-idx
		 &aux
		   (thread-idx thread-idx))))
  (thread-idx 0 :type fixnum)
  (cache-n 0 :type fixnum))

(defstruct (WaffeTensor (:print-function
			 (lambda (tensor stream depth)
			   (declare (ignore depth))
			   (format stream (render-tensor tensor))))
			(:constructor
			    sysconst
			    (value &key (backend *default-backend*)
				     (extend nil)
				     (thread-data nil)
				     (path-through-node? nil)
				     (no-jit nil)
			     &aux (data (init-waffe-tensor-data value))
			       (backend (check-backend backend extend))
			       (grad nil)
			       (thread-data thread-data)
			       (destructive? t)
			       (is-sysconst? t)
			       (force-ignore-jit no-jit)
			       (path-through-node? path-through-node?)
			       (is-mat (typep value 'mgl-mat:mat))
			       (grad-tmp (make-grad-tmp))))
	                (:constructor
			    tensor
			    (value &key (backend *default-backend*) (extend nil)
			     &aux (data (init-waffe-tensor-data value))
			       (is-ancestor-param t)
			       (is-mat (typep value 'mgl-mat:mat))
			       (is-param? t)
			       (backend (check-backend backend extend))
			       (grad `(nil nil))))
			(:constructor
			    const
			    (value &key (backend *default-backend*) (extend nil)
			     &aux (data (init-waffe-tensor-data value))
			       (backend (check-backend backend extend))
			       (is-mat (typep value 'mgl-mat:mat))
			       (grad nil)
			       (destructive? t))))
  "An structure of Waffe's Tensor.
This structure have:
@begin(enum)
@item(data (type of WaffeTensorContentType))
@item(the computation node for backprops, and grads)
@item(backend informations and parameters for optimizing.)
@end(enum)

There's three ways to make it.
@begin(deflist)
@term((const value))
@def(Constant tensor, grad won't be created.)
@term((tensor value))
@def(Parameter tensor, grad will be created.)
@term((sysconst value))
@def(Constant tensor where tensor sometime cached. Users don't have to use this.)
@end(deflist)

Value is following:
@begin(enum)
@item(simple-array)
@item(mgl-mat:mat (recommended))
@item(fixnum)
@item(float)
@item(null)
@item(cons)
@item(function (for lazy evaluation))
@item(ratio (when make, coerced to float))
@end(enum)

This structure is printable and printed nicely."
  (data nil :type WaffeTensorTypes)
  (grad-tmp (make-grad-tmp) :type grad-tmp)
  (backward nil :type boolean)
  (backend :mgl :type keyword)
  (grad nil :type WaffeTensorTypes)
  (variables nil :type list)
  state
  (is-mat nil :type boolean)
  (is-param? nil :type boolean)
  (is-ancestor-param nil :type boolean)
  (is-next-destruct? nil :type boolean)
  (destructive? nil :type boolean)
  (thread-data nil :type (or waffenodethread null))
  (is-sysconst? nil :type boolean)
  (path-through-node? nil :type boolean)
  (tensor-ident nil :type (or null symbol))
  (force-ignore-jit nil :type boolean)
  (key nil :type (or null cons))
  (idx nil :type (or null symbol))
  (is-data-destructed? nil :type boolean))

;(declaim (inline data
;		 (setf data)))
(defun data (tensor)
  "Access tensor's data. This won't be copied.

When tensor's data is lazy evaluted, this function behave following:
@begin(enum)
@item(When tensor is transposed and lazy evaluted, directly returns function object for speed.)
@item( When tensor is cached and lazy evaluted, returns mat object.)
@end(enum)

@begin(deflist)
@term(Input)
@def(WaffeTensor)
@term(Output)
@def(mgl-mat:mat, or waffetensorcontentdata)
@end(deflist)

when (data tensor) is a function and is:

@begin(deflist)
@term(cached mat)
@def(Return mgl-mat, this do not make copy)
@term(lazy-evaluation or transposed)
@def(Return function itself)
@end(deflist)

Note: this function is setfable and inlined"
  (declare (type waffetensor tensor))
  (typecase (waffetensor-data tensor)
    (function
     (let ((function-info
	     (funcall
	      (the
	       function
	       (waffetensor-data tensor))
	      tensor
	      nil
	      nil
	      nil
	      t)))
       (case function-info
	 (:lazy-eval (waffetensor-data tensor))
	 (:lazy-transpose (waffetensor-data tensor))
	 (T
	  (let ((result
		  (funcall
		   (the
		    function
		    (waffetensor-data tensor))
		   tensor
		   nil
		   nil
		   t)))
	    (if (null result)
		(the function
		     (waffetensor-data tensor))
		(the (values (or mat function) &optional) result)))))))
    (T (waffetensor-data tensor))))

(defun (setf data) (val &optional tensor)
  "Modifying tensor'data.

When the argument that you want to insert is a tensor, this function automatically reads tensor's data and modify with it"
  (if (typep val 'waffetensor)
      (setf (waffetensor-data tensor) (data val))
      (setf (waffetensor-data tensor) val)))

(defun double-random ()
  (let ((i (random 1.0)))
    (if (eq i 0.0)
	(setq i (double-random)))
    i))

(defun gaussiandb-random (var mean)
  (let* ((r (double-random))
	 (c (sqrt (* -2 (log r)))))
    (if (< (double-random) 0.5)
	(+    (* c
	      (sin (* 2.0 pi (double-random)))
	      var)
	      mean)
	(+    (* c
	      (cos (* 2.0 pi (double-random)))
	      var)))))

(declaim (ftype
	  (function
	   (waffetensorcontenttype)
	   WaffeTensorTypes)
	  init-waffe-tensor-data))
(defun init-waffe-tensor-data (content)
  ; todo: coerce: simple-array -> mgl-mat
  (declare (type waffetensorcontenttype content))
  (typecase content
    (ratio
     (if (eq mgl-mat:*default-mat-ctype* :double) ;...
	 (coerce content 'double-float)
	 (coerce content 'single-float)))
    (simple-array (mgl-mat:array-to-mat content))
    (T content)))

(defun check-backend (backend tensor)
  (if (null tensor)
      backend
      (waffetensor-backend tensor)))

(defmacro extend-from (new-tensor old-tensor)
  ; (extend-from (!randn `(10 10)) old-tensor) :backendとかを引き継ぐ
  (declare (ignore new-tensor old-tensor)))

(defmacro !allow-destruct (tensor)
  `(setf (waffetensor-is-next-destruct? ,tensor) t))

; is-tensor
(defun waffe-tensor-p (tensor)
  (typep tensor 'waffetensor))

(defmacro grad (tensor)
  "Accessing tensor's grad.

When tensor's grad is nil, an error occurs

@begin(deflist)
@term(Input)
@def(WaffeTensor)
@term(Output)
@def(An tensor's grad which is the type of mgl-mat:mat or waffetensorcontettype)
@end(deflist)

Note: grad is @b(not) setfable"
  `(progn
     (unless (typep ,tensor 'WaffeTensor)
       (error "The tensor is not a waffetensor."))
     
     (unless (waffetensor-grad ,tensor)
       (error "The tensor is not a parameter. Constants doesn't have a grad. If you need grad, please define it with (parameter (const ~~~))"))

     (if (typep (waffetensor-grad ,tensor) 'cons)
	 (error "A grad is nil. Please remain you need to call (backward out) before using a grad. Or, If you need grad, please define it with (parameter (const ~~~)). When using ~%~a" ,tensor))

     (waffetensor-grad ,tensor)))

(defmacro parameter (tensor)
  "Redefining new-tensor where old-tensor is const or tensor.

The new-tensor can made grads.

Excepted usage is like:
@begin[lang=lisp](code)
(setq my-param (parameter (!mul 0.01 (!randn `(10 10)))))
@end[lang=lisp](code)

Note that: tensor's computation node that old-tensor has, will be lost. Only tensor's data and backend will be extended.

@begin(deflist)
@term(Input)
@begin(def)
Tensor (as usual, defined by (const) (sysconst) (tensor))
@end(def)
@term(Output)
@begin(def)
Tensor (as usual, defined by (tensor))
@end(def)
@end(deflist)"
  `(with-slots ((data data) (backend backend)) ,tensor
     (tensor data :backend backend)))

(defun repeat-n (val n)
  (let ((a `(,val)))
    (dotimes (_ (1- n))
      (push val a))
    a))

(defun repeat-c (n &key (start 0))
  (let ((a `(,start))
	(i start))
    (dotimes (_ (1- n))
      (incf i 1)
      (push i a))
    (reverse a)))

(defmacro nth-var (tensor n)
  `(nth ,n (slot-value ,tensor 'variables)))

(defmacro nth-tensor (tensor n s)
  ; the nth variavle of tensor
  `(slot-value (nth-var ,tensor ,n) ,s))

(defmacro setfgradtmp (tensor value)
  `(progn
     (setf (grad-tmp-grad-called (waffetensor-grad-tmp ,tensor)) t)
     (setf (grad-tmp-value (waffetensor-grad-tmp ,tensor)) ,value)))

(defun backward (tensor)
  "Compute back propagation by traversing the Tensor's computation node.

The parameters of the model defined by (tensor) or to which (Parameter tensor) is applied, store the gradient in grad slot.

Note that: tensor must be the shape of `(1) or single value. Otherwise an error occurs.

In the process calculating backward, new backwards won't be created. (*no-grad* automatically becomes t)

@begin(deflist)
@term(Input)
@def(WaffeTensor)
@term(Output)
@def(NIL)
@end(deflist)"
  (declare (type waffetensor tensor))
  (if (typep (data tensor) 'mgl-mat:mat)
      (unless (eq (!shape tensor) `(1))
	(error "grad can be implicitly created only for scalar outputs")))

  (setq *no-grad* t)
  (backward1 tensor)
  (setq *no-grad* nil)
  nil)

(declaim (inline step-next-node))

(defun step-next-node (tensor n)
  (if (waffetensor-is-ancestor-param (nth-var tensor n))
      (backward1 (nth-var tensor n))))

(declaim (ftype (function (waffetensor) null) backward1))
(defun backward1 (tensor)
  (declare (optimize (speed 3) (space 0) (safety 0))
	   (type waffetensor tensor))
  (cond
    ((waffetensor-backward tensor) ;Backward exists?
      (let* ((grad-tmp-before (waffetensor-grad-tmp tensor))
	     (grad-before (if (grad-tmp-grad-called grad-tmp-before) ;check if the node is a top
			      (grad-tmp-value grad-tmp-before)
			      (sysconst 1))))
	; calculating backward(state, dy) -> x.grad, y.grad...

	(progn
	  (let ((grads (funcall
			(the function
			     (call-backward (waffetensor-state tensor)))
			grad-before)))
	    (declare (type list grads)) ; Todo: Print Error
	    (unless (= (length (waffetensor-variables tensor))
		       (length grads))
	      (error "backward error: The number of :forward args doesnt correspond with of :backward"))
	    
	    (dotimes (n (length grads))
	      (setf (waffetensor-thread-data (nth n grads))
		    (waffetensor-thread-data tensor))
	      (setfgradtmp (nth-var tensor n) (nth n grads))
	      (step-next-node tensor n)))
	  nil)))
    (T
     ; Collecting :grad-tmp and copying them to: grad
     (when (waffetensor-grad tensor)
	   ; the tensor is the end of node.
       (when (grad-tmp-value (waffetensor-grad-tmp tensor))
	   ; is grad-tmp already created?
	 (if (typep (waffetensor-grad tensor) 'cons)
	     ; is it first value? or not?
	     (let ((new-grad (grad-tmp-value (waffetensor-grad-tmp tensor))))
	       (setf (waffetensor-grad tensor) (value new-grad)))
	     
	     ;Otherwise (grad-tmp is created), Sum up grads for multiple variables
	     (if (typep (waffetensor-grad tensor) 'mat)
		 (axpy! 1.0 ; todo: integrate add with jit.
			(value (grad-tmp-value
			       (waffetensor-grad-tmp tensor)))
			(waffetensor-grad tensor))
		 (setf (waffetensor-grad tensor)
		       (+ (the single-float (waffetensor-grad tensor))
			  (the single-float
			       (value (grad-tmp-value
				 (waffetensor-grad-tmp tensor))))))))))))
  nil)

(declaim (ftype (function (cons) waffetensor) !zeros !ones))
(defun !zeros (shape)
  "Initializing constant tensor with given shape, where initial elements are zero.

Input: shape (cons)

Output: Tensor (which is constant)

Example:
@begin[lang=lisp](code)
(!zeros `(10 10))
;#Const(((0.0 0.0 ~ 0.0 0.0)        
;                ...
;        (0.0 0.0 ~ 0.0 0.0)) :mgl t :shape (10 10))
@end[lang=lisp](code)"
  (declare (type cons shape))
  (const (mgl-mat:make-mat shape :initial-element 0)))

(defun !ones (shape)
  "The same as !zeros but initial element is one.

Example:
@begin[lang=lisp](code)
(!ones `(10 10))
;#Const(((1.0 1.0 ~ 1.0 1.0)        
;                ...
;        (1.0 1.0 ~ 1.0 1.0)) :mgl t :shape (10 10))
@end[lang=lisp](code)"
  (declare (type cons shape))
  (const (mgl-mat:make-mat shape :initial-element 1)))

(defun !fill (shape element)
  "The same as !zeros, !ones but initial element is given element.

Note: the argument @cl:param(element) coerced into @cl:param(mgl-mat:*default-mat-ctype*)

Example:
@begin[lang=lisp](code)
(!fill '(10 10) 10)
;#Const(((10.0 10.0 ~ 10.0 10.0)        
;                  ...
;        (10.0 10.0 ~ 10.0 10.0)) :mgl t :shape (10 10))
@end[lang=lisp](code)
"
  (declare (type cons shape))
  (const (mgl-mat:make-mat shape :initial-element element)))

(defmacro !arange (&rest args)
  "Like numpy's arange, arange can be called with a varying number of positional arguments:

@begin(section)
@title((!arange stop))
@begin[lang=lisp](code)
(!arange 10)
;#Const((0.0 1.0 ~ 8.0 9.0) :mgl t :shape (10))
@end[lang=lisp](code)
@end(section)

@begin(section)
@title((!arange start stop))
@begin[lang=lisp](code)
(!arange 3 10)
;=>#Const((3.0 4.0 ~ 8.0 9.0) :mgl t :shape (7))
@end[lang=lisp](code)
@end(section)

@begin(section)
@title((!arange start stop step))
@begin[lang=lisp](code)
(!arange 3 10 2)
;#Const((3.0 5.0 7.0 9.0) :mgl t :shape (4))
@end[lang=lisp](code)
@end(section)"
  `(let ((base-array (numcl:arange ,@args)))
     (const (make-mat (numcl:shape base-array)
		      :initial-contents base-array))))

(defmacro !copy (tensor &aux (new-tensor (gensym)))
  `(let ((,new-tensor (!zeros-like ,tensor)))
     (mgl-mat:copy! (data ,tensor) (data ,new-tensor))
     ,new-tensor))

(declaim (ftype (function (waffetensor fixnum fixnum) waffetensor) !set-batch))
(defun !set-batch (dataset start-row-index batch-size)
  "Set batch where dataset is a 2d mat.

Todo: Backward."
  
  (declare (optimize (speed 3) (space 0) (safety 0))
	   (type waffetensor dataset)
	   (type fixnum start-row-index batch-size))
  (let ((dim (the fixnum (mgl-mat:mat-dimension (data dataset) 1))))
    (mgl-mat:reshape-and-displace! (data dataset)
                           (list batch-size dim)
                           (the fixnum (* start-row-index dim)))
    dataset))

(defun !reset-batch (dataset)
  "Reset batch of dataset (i.e.: reset dataset's displacement)"
  (let* ((dim (mgl-mat:mat-dimension (data dataset) 1))
         (len (/ (mgl-mat:mat-max-size (data dataset)) dim)))
    (reshape-and-displace! (data dataset) (list len dim) 0)
    dataset))

(defun !aref (tensor &rest dims)
  "Very fast aref.

This function is setfable.

Cuts the area specified by dims from Tensor and generates a new Const.

This function creates computation node.

dims are following:
@begin(enum)
@item(fixnum)
@item(t ... which means 0 ... maxlen)
@begin(item)
Cons (e.g. '(1 3) reads 1<=x<3)
@end(item)
@end(enum)

Example:
@begin[lang=lisp](code)

(setq a (!randn `(10 10)))

;=> #Const(((0.280... 1.941... ~ 0.723... -0.47...)        
;                   ...
;          (-1.01... 0.232... ~ -1.16... 0.405...)) :mgl t :shape (10 10))

(!aref a '(0 3) t)
(!aref a t '(0 3))
(!aref a 1 1)
(setq a (setf (!aref a '(0 3) '(0 3)) (!zeros '(3 3)))) ; to update nodes

@end[lang=lisp](code)"
  (call (ArefTensor dims) tensor))

(defun !areflist (tensor dims)
  (call (ArefTensor dims) tensor))

(defun (setf !aref) (value tensor &rest dims)
  "Todo: Define it as macro and (setq tensor ~)"
  (setf tensor (setf (!areflist tensor dims) value)))

(defun (setf !areflist) (value tensor dims)
  ; For backward, you need to call it like (setq z (setf (!aref x ~) ~))
  ; To solve this problem, i guess i need more macros.
  (setf tensor (call (SetfArefTensor dims) tensor (assure-tensor value))))

(defmacro !row-major-aref (tensor index)
  `(mgl-mat:row-major-mref (data ,tensor) ,index))

(defmacro !with-mgl-operation (tensor var &body body)
  `(let ((,var (data ,tensor)))
     ,@body))

(defun !random (dims limit)
  "Initialize an tensor of dims (cons)

!random can be called with a varying number of type of arguments:

@begin(section)
@title(When limit=fixnum)
init within the range of @c(0<=x<limit)

@begin[lang=lisp](code)
;#Const(((1.0 2.0 ~ 2.0 1.0)        
;                 ...
;        (2.0 2.0 ~ 2.0 2.0)) :mgl t :shape (10 10))
@end[lang=lisp](code)
@end(section)

@begin(section)
@title(When limit=single-float)
init within the range of @c(0<=x<limit)
@begin[lang=lisp](code)
(!random '(10 10) 3.0)
;#Const(((0.152... 2.203... ~ 2.360... 2.216...)        
;                 ...
;        (1.003... 2.257... ~ 2.305... 2.025...)) :mgl t :shape (10 10))
@end[lang=lisp](code)
@end(section)

@begin(section)
@title(When limit=(cons single-float1 single-float2))
init with single-float1<=x<single-float2, where each element is single-float.
@begin[lang=lisp](code)
(!random '(10 10) '(1.0 3.0))
;#Const(((1.982... 1.526... ~ 1.388... 1.312...)        
;                 ...
;        (1.829... 2.676... ~ 1.226... 2.980...)) :mgl t :shape (10 10))
@end[lang=lisp](code)
@end(section)

Return: WaffeTensor
"
  (let* ((res (!zeros dims))
         (upper-limit (if (listp limit) (second limit) limit))
         (lower-limit (if (listp limit) (first limit) 0))
         (len (if (listp dims) (reduce #'* dims) dims))
         (tmp-limit (- upper-limit lower-limit)))
    (loop for n from 0 to (1- len)
          do (setf (!row-major-aref res n)
                   (+ (random tmp-limit) lower-limit)))
    res))

(declaim (ftype (function ((or cons fixnum) function) waffetensor) !random-with))
(defun !random-with (dims f)
  "Initializes the tensor of dims. Each element is initialized with @cl:param(f) where f is a lambda exp and called with index.

Warning: Using mref and slow algorithm, @b(it is so slow).

Example:
@begin[lang=lisp](code)
(!random-with '(10 10) #'(lambda (n) n))
;#Const(((0.0 1.0 ~ 8.0 9.0)        
;                 ...
;        (90.0 91.0 ~ 98.0 99.0)) :mgl t :shape (10 10))
@end[lang=lisp](code)

See also: !init-with which is alias for !random-with.
"
  (declare (optimize (speed 3) (safety 0) (space 0))
	   (type function f))
  (let* ((res (make-array dims :initial-element 0))
         (len (the fixnum (if (listp dims) (reduce #'* dims) dims))))
    (loop for n fixnum from 0 to (1- len)
          do (setf (row-major-aref res n)
                   (funcall f n)))
    (const res)))

(declaim (inline !init-with))
(defun !init-with (dims f)
  "Alias for !random-with. This function is inlined."
  (!random-with dims f))

(defun !normal (dims &optional (mean 2.0) (var 1.0))
  "Init with normal distribution.

Warning: Using mref and slow algorithm, @b(its sooo slow.)

It is recommended to use !randn and transform it instead."
  (let* ((res (!zeros dims))
         (len (if (listp dims) (reduce #'* dims) dims)))
    (loop for n from 0 to (1- len)
          do (setf (!row-major-aref res n) (gaussiandb-random var mean)))
    res))

(defun !randn (dims)
  "Initializes tensor with normal distribution in a faster way where mean=0.0, var=1.0.

Example:

@begin[lang=lisp](code)
(!randn `(10 10))
;#Const(((0.677... 0.054... ~ 0.257... 0.261...)        
;                 ...
;        (0.063... 0.607... ~ 0.460... 0.730...)) :mgl t :shape (10 10))
@end[lang=lisp](code)"
  (const (uniform-random! (make-mat dims))))

(defun !beta (dims alpha beta)
  "Initializes tensor with samples of beta distribution in a faster way.

Algorithm: https://dl.acm.org/doi/pdf/10.1145/359460.359482

x=[0,1]

a = min(alpha, beta)

b = max(alpha, beta)

PDF: fX(x)=x^a−1*(1−x)*b−1/B(a,b)

where B(a,b)=∫1,0{x^a−1(1−x)^b−1}dx

@begin[lang=lisp](code)
(time (!beta '(200) 5.0 1.0))
;Evaluation took:
;  0.000 seconds of real time
;  0.000063 seconds of total run time (0.000063 user, 0.000000 system)
;  100.00% CPU
;  143,846 processor cycles
;  0 bytes consed
  
;#Const((0.813... 0.832... ~ 0.865... 0.787...) :mgl t :shape (200))
@end[lang=lisp](code)"

  (declare (optimize (speed 3))
	   (type cons dims)
	   (type single-float alpha beta))
  (let* ((a (min alpha beta))
 	 (b (max alpha beta))
	 (result (!zeros dims))
	 (size (!size result)))
    (declare (type fixnum size))
    (with-facet (array ((data result) 'backing-array :direction :io))
      (declare (type (simple-array single-float) array))
      ; Todo For GPU.
      (loop for i fixnum upfrom 0 below size
	    do (setf (aref array i)
		     (if (> a 1)
			 (!beta-bb alpha a b)
			 (!beta-bc alpha a b)))))
    result))

(declaim (ftype (function
		 (single-float single-float single-float)
		 single-float)
		!beta-bb
		!beta-bc))
(defun !beta-bb (a0 a b)
  "Generates beta variances.

Algorithm: https://dl.acm.org/doi/pdf/10.1145/359460.359482

Note: !beta excepts that @c((min a b) > 1)"
  (declare (optimize (speed 3) (safety 0) (debug 0))
	   (type cons dims)
	   (type single-float a0)
	   (type (single-float 0e0) a b))

  (unless (> (min a b) 1.0)
    (error "cl-waffe:!beta failed because of (min a b) > 1."))

  (let* ((alpha (+ a b))
  	 (beta  (sqrt (the (single-float 0e0)
			   (/ (- alpha 2.0)
			      (- (* 2.0 a b) alpha)))))
	 (gamma (+ a (/ beta)))
	 (r0 0.0)
	 (w0 0.0)
	 (t0 0.0))
    (labels ((next (&aux
		      (u1 (random 1.0))
		      (u2 (random 1.0))
		      (v (* beta (- (log u1) (log (+ 1.0 (- u1)))))))
	       (declare (type single-float u1 u2 v))
	       
	       (setq w0 (* a (exp v)))
	       (setq r0 (- (* gamma v) 1.3862944))
	       
	       (let* ((z (* u1 u1 u2))
		      (s (+ a r0 (- w0))))
		 (declare (type single-float z s))
		 
		 (if (>= (+ s 2.609438) (* 5 z))
		     nil
		     (progn
		       (setq t0 (log z))
		       (if (>= s t0)
			   nil
			   t))))))
      (loop while (and
		   (next)
		   (< (+ r0
			 (* alpha (- (log alpha) (log (+ b w0)))))
		      t0)))

      (if (= a a0)
	  (/ w0 (+ b w0))
	  (/ b (+ b w0))))))


(defun !beta-bc (a0 a b)
  "Generates beta variances.

Algorithm: https://dl.acm.org/doi/pdf/10.1145/359460.359482

Note: !beta excepts that @c((min a b) <= 1)"
  (declare (optimize (speed 3) (safety 0) (debug 0))
	   (type cons dims)
	   (type single-float a0)
	   (type (single-float 0e0) a b))

  (unless (<= (min a b) 1.0)
    (error "cl-waffe:!beta failed because of (min a b) <= 1."))

  (let* ((alpha (+ a b))
  	 (beta  (/ b))
	 (gamma (+ 1 a (- b)))
	 (k1 (* gamma (+ 0.0138889 (* 0.0416667 b)) (/ (+ (* a b)
							  -0.777778))))
	 (k2 (+ 0.25 (* b (+ 0.5 (/ 0.258 gamma)))))
	 (z  0.0)
	 (y  0.0)
	 (v 0.0)
	 (w 0.0)
	 (f t)
	 (u1 0.0)
	 (u2 0.0))
    (declare (type single-float alpha beta gamma k1 k2 z y w v u1 u2))
    
    (labels ((next ()
	     (setq u1 (random 1.0))
	     (setq u2 (random 1.0))
	     (if (>= u1 0.5)
		 (progn
		   (setq z (* u1 u1 u2))
		   (if (<= z 0.25)
		       (progn
			 (setq v (* beta
				    (the single-float
					 (log (the (single-float 0e0)
						   (/ u1 (- 1 u1)))))))
			 (setq w (* a (exp v)))
			 nil)
		       (if (>= z k2)
			   t
			   nil)))
		 (progn
		   (setq y (* u1 u2))
		   (setq z (* u1 y))
		   (if (>= (+ (* 0.225 u2) z (- y))
			   k1)
		       t
		       nil)))))

      (loop while (and f (next))
	    do (progn
		 (setq v (* beta (log (the (single-float 0e0) (/ u1 (- 1 u1))))))
		 (setq w (* a (exp v)))

		 (if (< (- (* alpha
			      (log (the (single-float 0e0) (/ a (+ b w)))))
			   1.3862944)
			(log z))
		     (setq f nil))))

      (if (= a a0)
	  (/ w (+ b w))
	  (/ b (+ b w))))))

(defun !gamma (dims k &optional (theta 1.0))
  "Initialize tensor with samples of gamma distribution.

Todo: Use fast algorithms and approximations in response to @cl:param(k).

Example:
@begin[lang=lisp](code)
(!gamma '(10 10) 1.0)
;#Const(((2.155... 3.374... ~ 1.274... 0.147...)        
;                 ...
;        (0.194... 0.081... ~ 0.816... 0.209...)) :mgl t :shape (10 10))
@end[lang=lisp](code)"
  (declare ;(optimize (speed 3))
	   (type cons dims)
	   (type single-float scale))
  
  ; ↓やる気無くした人 適当な早いアルゴリズム実装してぇ~~
  (const (make-mat dims
		   :initial-contents (numcl:gamma k theta dims))))

(defun !chisquare (dims df)
  "@b(Not implemented yet)
Todo: Use fast algorithms and approximations.

Example:
@begin[lang=lisp](code)
@end[lang=lisp](code)"
  (declare (ignore df dims))
  (error "Not Implemented."))

(defun !bernoulli (dims rate)
  "Init a tensor of dims with bernoulli

rate is single-float, and [0 1]

See also: @cl:param(!binomial), alias for it.

Example:
@begin[lang=lisp](code)
(!binomial '(10 10) 0.5)
;#Const(((1.0 0.0 ~ 1.0 1.0)        
;                 ...
;        (0.0 1.0 ~ 1.0 0.0)) :mgl t :shape (10 10))
@end[lang=lisp](code)"
  (declare (optimize (speed 3))
	   (type cons dims)
	   (type (single-float 0e0) rate))
  (unless (<= rate 1.0)
    (error "!bernoulli: rate must be in the range of [0 1]"))
  (!modify (!zeros dims) :bernoulli (const rate)))

(declaim (inline !binomial))
(defun !binomial (dims rate)
  "Alias for !bernoulli"
  (!bernoulli dims rate))

(defun !shape (tensor &optional (nth nil))
  "Returns the shape of tensor when nth=nil.

@cl:param(nth) indicates the index of shape, !shape return specified value.

Example:
@begin[lang=lisp](code)
(setq a (!randn `(10 10 10)))
(!shape a) ; => (10 10 10)
(!shape a 0) ;=> 10
@end[lang=lisp](code)"
  (declare (type waffetensor tensor))
  (unless (typep (waffetensor-data tensor) 'waffe-array)
    (unless (or (typep (waffetensor-data tensor) 'function)
		(typep (waffetensor-data tensor) 'compiled-function))
      (return-from !shape `(0))))
    
  (if (not (null nth))
      (let* ((n (if (typep nth 'waffetensor)
	 	    (waffetensor-data nth)
		    nth))
	     (n (if (< n 0) (+ (!dims tensor) n) n)))
	(typecase (waffetensor-data tensor)
	  (function
	   (nth n (funcall (waffetensor-data tensor) tensor t nil)))
	  (T
	   (mat-dimension (waffetensor-data tensor) n))))
      (typecase (waffetensor-data tensor)
	(function
	 (funcall (waffetensor-data tensor) tensor t nil))
	(T
	 (mat-dimensions (waffetensor-data tensor))))))

(declaim (ftype (function (waffetensor) fixnum) !dims !size))
(defun !dims (tensor)
  "Returns the total length of a given tensor's dims

Example:
@begin[lang=lisp](code)
(!dims (!zeros '(10 10 10))) ; => 3
@end[lang=lisp](code)"
  (the fixnum (length (!shape tensor))))

(defun !size (tensor)
  "Returns the total size of a tensor

Example:
@begin[lang=lisp](code)
(!size (!zeros '(10 10 10))) ; => 1000
@end[lang=lisp](code)"
  (apply #'* (!shape tensor)))

(defun !size-1 (tensor)
  (1- (!size tensor)))

(defun !zeros-like (tensor)
  "Return a const where the shape is the same as tensor but elements are zero.

Example:
@begin[lang=lisp](code)
(setq a (!randn `(10 10)))
(!zeros-like a)
;#Const(((0.0 0.0 ~ 0.0 0.0)        
;                 ...
;        (0.0 0.0 ~ 0.0 0.0)) :mgl t :shape (10 10))
@end[lang=lisp](code)"
  (!zeros (!shape tensor)))

(defun !ones-like (tensor)
  "Return a const where the shape is the same as tensor but elements are one.
Example:
@begin[lang=lisp](code)
(setq a (!randn `(10 10)))
(!ones-like a)
;#Const(((1.0 1.0 ~ 1.0 1.0)        
;                 ...
;        (1.0 1.0 ~ 1.0 1.0)) :mgl t :shape (10 10))
@end[lang=lisp](code)"
  (!ones (!shape tensor)))

(defun !full-like (tensor element)
  "Return a const where the shape is the same as tensor but elements are specified value by @cl:param(element).
Example:
@begin[lang=lisp](code)
(setq a (!randn `(10 10)))
(!full-like a 3)
;#Const(((3.0 3.0 ~ 3.0 3.0)        
;                 ...
;        (3.0 3.0 ~ 3.0 3.0)) :mgl t :shape (10 10))
@end[lang=lisp](code)"
  (!fill (!shape tensor) element))

(defmacro detach (tensor)
  "Create a Const with all information except data and backend erased.

This macro expanded to @c((const (data tensor))).

Note: this macro doesn't clone data itself.

Example:
@begin[lang=lisp](code)
(setq a (parameter (!randn `(10 10))))
;#Parameter{((0.062... 0.716... ~ 0.088... 0.692...)            
;                         ...
;            (0.458... 0.194... ~ 0.902... 0.480...)) :mgl t :shape (10 10) :device :MGL :backward NIL}
(detach a)
;#Const(((0.062... 0.716... ~ 0.088... 0.692...)        
;                 ...
;        (0.458... 0.194... ~ 0.902... 0.480...)) :mgl t :shape (10 10))
@end[lang=lisp](code)"
  `(const (data ,tensor)))

(defun !where (condition tensor then else)
  "Return a tensor of elements selected from either x or y, depending on condition.

@cl:param(condition) is given as a lambda expression, which called with an value of (aref tensor index).

!where defined as

@c(out = if (condition(tensor[i]), then, else))

Return: A tensor of shape that equal to the condition.

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!random `(10 10) '(-1.0 1.0)))
;#Const(((0.042... -0.36... ~ 0.250... 0.967...)        
;                 ...
;        (-0.21... 0.962... ~ -0.32... 0.215...)) :mgl t :shape (10 10))

(!where #'(lambda (x) (> x 0)) a 1.0 0.0)
;#Const(((1.0 0.0 ~ 1.0 1.0)        
;                 ...
;        (0.0 1.0 ~ 0.0 1.0)) :mgl t :shape (10 10))

; works as ReLU

(!mul a (!where #'(lambda (x) (> x 0)) a 1.0 0.0))
;#Const(((0.042... 0.0... ~ 0.250... 0.967...)        
;                 ...
;        (0.0... 0.962... ~ 0.0... 0.215...)) :mgl t :shape (10 10))
@end[lang=lisp](code)
@end(section)"
  (declare (optimize (speed 3))
	   (type function condition)
	   (type waffetensor tensor)
	   (type single-float then else))
  (value tensor)	      
  (let ((result (!zeros (!shape tensor))))
    (with-facets ((result-array ((data result)
				 'backing-array
				 :direction
				 :output))
		  (tensor-array ((data tensor)
				 'backing-array
				 :direction
				 :input)))
      (declare (type (simple-array single-float) result-array tensor-array))
      (loop for i fixnum upfrom 0 below (!size tensor)
	    do (setf (aref result-array i)
		     (if (funcall condition (aref tensor-array i))
			 then
			 else))))
    result))

(defun !index () "Todo")

(defun write-description (res backward backend)
  ; Parameter { ... <= here }
  (write-string (format nil " :device :~a :backward ~A" backend backward) res))

(defun reduce-str (obj)
  ; align string content of tensor following *print-char-max-len*
  ; Todo: 1.00000000001d0 <- 末端も表示
  (let ((str (format nil "~a" obj)))
    (if (>= (length str) *print-char-max-len*)
	(concatenate 'string (subseq str 0 *print-char-max-len*) "...")
	str)))

(defmacro !aref-array (array &rest args) ; possibly too slow...
  `(let ((res (data (!aref (const (mgl-mat:array-to-mat ,array)) ,@args))))
     (mgl-mat:mat-to-array (mgl-mat:reshape! res (cdr
						  (mgl-mat:mat-dimensions res)))))) ;unsqueeze

(defun !unsqueeze-array (array)
  (mgl-mat:mat-to-array (mgl-mat:reshape!
			 (mgl-mat:array-to-mat array) (cdr (array-dimensions array)))))

(defun pprint-1d-vector (stream data)
  (if (> (length (array-dimensions data)) 1)
      (error ""))
  (if (>= (apply #'* (array-dimensions data)) *print-arr-max-size*)
      (write-string (format nil "(~A ~A ~2~ ~A ~A)" ; todo: i wanna display last 3 digits.
			    (reduce-str (aref data 0))
			    (reduce-str (aref data 1))
			    (reduce-str (aref data (-  (length data) 2)))
			    (reduce-str (aref data (1- (length data)))))
		    stream)
      (progn
	(write-string "(" stream)
	(dotimes (i (apply #'* (array-dimensions data)))
	  (write-string (format nil "~A" (reduce-str (aref data i))) stream)
	  (unless (= i (1- (apply #'* (array-dimensions data))))
	    (write-string " " stream)))
	(write-string ")" stream))))

(defun pprint-vector (stream data &optional (newline T) (indent-size 0))
  (cond
    ((= 1 (length (array-dimensions data)))
     (pprint-1d-vector stream data))
    ((= 1 (car (array-dimensions data)))
     (write-string "(" stream)
     (pprint-vector stream (!unsqueeze-array data) newline (1+ indent-size))
     (write-string ")" stream))
    (T
     (write-string "(" stream)
     (if (< (car (array-dimensions data)) *print-mat-max-size*)
	 (let ((fd (car (array-dimensions data))))
	   (dotimes (i fd)
	     (pprint-vector stream (!aref-array data i) newline (1+ indent-size))
	     (unless (= i (1- fd))
	       (if newline
		   (progn
		     (write-char #\Newline stream)
		     (dotimes (k (1+ indent-size))
		       (write-string " " stream)))
		   (write-string " " stream))))
	   (write-string ")" stream))
	 (progn
	   (labels ((render-v (line do-newline)
		      (pprint-vector stream line newline (1+ indent-size))
		      (if do-newline
			  (if newline
				(dotimes (k (1+ indent-size))
				  (write-string " " stream))))))
	     (render-v (!aref-array data 0) T)
	     ;(render-v (numcl:aref data 1) T)
	     (if newline
		 (progn
		   (write-char #\newline stream)
		   (dotimes (_ (+ (* 2 indent-size) 3))
		     (write-string " " stream))
		   (write-string "..." stream)
		   (write-char #\newline stream)
		   (dotimes (k (1+ indent-size))
		     (write-string " " stream))))
	     ;(render-v (numcl:aref data (- (car (numcl:shape data)) 2)) T)
	     (render-v (!aref-array data (1- (car (array-dimensions data)))) NIL)
	     (write-string ")" stream)))))))

(defun render-tensor (tensor &optional (newline T) (indent-size 0))
  (with-slots ((contents data) (backward backward) (backend backend) (grad grad)) tensor
    (with-output-to-string (res)
      (if (null grad)
	  (write-string "#Const(" res)
	  (write-string "#Parameter{" res))
      
      (if (or (typep contents 'array)
	      (typep contents 'vector))
	  (progn
	    (pprint-vector res contents newline (if (null grad)
						    (+ indent-size (length "#Const("))
						    (+ indent-size (length "#Parameter{"))))
	    (write-string (format nil " :mgl nil :shape ~a" (!shape contents)) res)
	    (unless (null grad)
	      (write-description res backward backend))
	    (if (null grad)
		(write-string ")" res)
		(write-string "}" res)))
	  (if (typep contents 'mgl-mat:mat)
	      (progn
		(pprint-vector res (mgl-mat:mat-to-array contents) newline
							 (if (null grad)
							     (+ indent-size (length "#Const("))
							     (+ indent-size (length "#Parameter{"))))
		(write-string (format nil " :mgl t :shape ~a" (mgl-mat:mat-dimensions contents)) res)
		(unless (null grad)
		  (write-description res backward backend))
		(if (null grad)
		    (write-string ")" res)
		    (write-string "}" res)))
	      (progn
		(write-string (format nil "~A" contents) res)
		(unless (null grad)
		  (write-description res backward backend))
		(if (null grad)
		    (write-string ")" res)
		    (write-string "}" res)))))
      res)))

