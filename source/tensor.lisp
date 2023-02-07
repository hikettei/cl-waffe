
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
			     &aux (data (init-waffe-tensor-data value))
			       (backend (check-backend backend extend))
			       (grad nil)
			       (thread-data thread-data)
			       (destructive? t)
			       (is-sysconst? t)
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
  (key nil :type (or null cons))
  (idx nil :type (or null symbol))
  (is-data-destructed? nil :type boolean))

(declaim (inline data
		 (setf data)))
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

Note: this function is setfable and inlined"
  (declare (type waffetensor tensor))
  (typecase (waffetensor-data tensor)
    (function
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
	   (the (values (or mat function) &optional) result))))
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
	 (coerce content 'float)))
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
	      (setfgradtmp (nth-var tensor n) (nth n grads)))

	    (dotimes (n (length grads))
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
	       (setf (waffetensor-grad tensor) (data new-grad)))
	     
	     ;Otherwise (grad-tmp is created), Sum up grads for multiple variables
	     (if (typep (waffetensor-grad tensor) 'mat)
		 (axpy! 1.0
			(data (grad-tmp-value
			       (waffetensor-grad-tmp tensor)))
			(waffetensor-grad tensor))
		 (setf (waffetensor-grad tensor)
		       (+ (the single-float (waffetensor-grad tensor))
			  (the single-float
			       (data (grad-tmp-value
				 (waffetensor-grad-tmp tensor))))))))))))
  nil)

(declaim (ftype (function (cons) waffetensor) !zeros !ones))
(defun !zeros (shape)
  "Initializing constant tensor with given shape, where initial elements are zero.

Input: shape (cons)

Output: Tensor (which is constant)

Example:
@begin[lang=lisp](code)
(!zeros `(10 10)) => #Const(((0.0 0.0 ~ 0.0 0.0)        
                 ...
       (0.0 0.0 ~ 0.0 0.0)) :mgl t :shape (10 10))
@end[lang=lisp](code)"
  (const (mgl-mat:make-mat shape :initial-element 0)))

(defun !ones (shape)
  "The same as !zeros but initial element is one"
  (const (mgl-mat:make-mat shape :initial-element 1)))

(defun !fill (shape element)
  "The same as !zeros, !ones but initial element is given element
Input: element ... fixnum or single-float"
  (const (mgl-mat:make-mat shape :initial-element element)))

(defmacro !arange (&rest args)
  "arange can be called with a varying number of positional arguments:"
  `(const (mgl-mat:make-mat (numcl:shape (numcl:arange ,@args))
			    :initial-contents (numcl:arange ,@args))))

(defmacro !copy (tensor &aux (new-tensor (gensym)))
  `(let ((,new-tensor (!zeros-like ,tensor)))
     (mgl-mat:copy! (data ,tensor) (data ,new-tensor))
     ,new-tensor))

(declaim (ftype (function (waffetensor fixnum fixnum) waffetensor) !set-batch))
(defun !set-batch (dataset start-row-index batch-size)
  "Set batch where dataset is a 2d mat

Note: dataset's shape must be divided by batch-size (Todo Fix)

Internally, this function just modifying dataset's displacement.

So it's very fast."
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

@end[lang=lisp](code)

Todo: Bugfix"
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
	 
(defmacro !where ()) ; todo
(defmacro !index ()) ; todo

(defmacro !row-major-aref (tensor index)
  `(mgl-mat:row-major-mref (data ,tensor) ,index))

(defmacro !with-mgl-operation (tensor var &body body)
  `(let ((,var (data ,tensor)))
     ,@body))

(defun !random (dims limit)
  "Initialize an tensor of dims (cons)

!random can be called with a varying number of type of arguments:

When limit=fixnum, init with 0~fixnum

When limit=single-float, init with 0~single-float

When limit=(cons fixnum1 fixnum2), init with fixnum1~fixnum2, where each element is fixnum

When limit=(cons single-float1 single-float2), init with single-float1~single-float2, where each element is single-float

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
  "Initialize the tensor of dims.

f is function and each element is initialized with f's value."
  (declare (optimize (speed 3) (safety 0) (space 0))
	   (type function f))
  (let* ((res (make-array dims :initial-element 0))
         (len (the fixnum (if (listp dims) (reduce #'* dims) dims))))
    (loop for n fixnum from 0 to (1- len)
          do (setf (row-major-aref res n)
                   (funcall f)))
    (const res)))

(defun !normal (dims &optional (mean 2.0) (var 1.0))
  "Init with normal."
  (let* ((res (!zeros dims))
         (len (if (listp dims) (reduce #'* dims) dims)))
    (loop for n from 0 to (1- len)
          do (setf (!row-major-aref res n) (gaussiandb-random var mean)))
    res))

(defmacro !randn (dims) ; this can be rewrited it by mgl-mat
  "Init with normal where mean=0.0, var=1.0"
  `(!normal ,dims 0.0 1.0))

(defun !beta (dims a b)
  "TODO"
  (declare (ignore dims a b)))

(defun !gamma (dims scale)
  "TODO"
  (declare (ignore dims scale)))

(defun !chisquare (dims df)
  "TODO"
  (declare (ignore dims df)))

(defun !bernoulli (dims rate)
  "Init a tensor of dims with bernoulli

rate is single-float, and [0 1]"
  (!modify (!zeros dims) :bernoulli (const rate)))

(defun !shape (tensor &optional (nth nil))
  "Return the shape of tensor

Tensor is WaffeTensor.

Output is Cons or fixnum

When nth is not nil, return (nth nth (!shape tensor))"
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

(defun !dims (tensor)
  "Return total length of the given tensor's dims

Example:
@begin[lang=lisp](code)
(!dims (!zeros '(10 10 10))) ; => 3
@end[lang=lisp](code)"
  (the fixnum (length (!shape tensor))))

(defun !size (tensor)
  "Return total size of tensor

Example:

@begin[lang=lisp](code)
(!dims (!zeros '(10 10 10))) ; => 1000
@end[lang=lisp](code)"
  (apply #'* (!shape tensor)))

(defun !size-1 (tensor)
  (1- (!size tensor)))

(defun !zeros-like (tensor)
  "Return a const where the shape is the same as tensor but elements are zero."
  (!zeros (!shape tensor)))

(defun !ones-like (tensor)
  "Return a const where the shape is the same as tensor but elements are one."
  (!ones (!shape tensor)))

(defun !full-like ()
  "fulls like(todo)")

(defmacro detach (tensor)
  "Expanded to (const (data tensor)).

Note: this macro doesn't clone data itself"
  `(const (data ,tensor)))

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

