
(in-package :cl-waffe)

#|
Here's utils for tensor.
1. Displaying tensors
2. Initializing tensors with sampling specified distributions.
3. Backward function
4. Utils for tensors. 
|#


(defparameter *no-grad* nil
  "When t, some node will be ignored. see references below for details. default: nil")

(defparameter *verbose* nil "When t, all computation nodes will be displayed to stream t")

(defparameter *single-node* nil "This parameter becames t when backward, the size of x.variables is 1.")

(defparameter *backward-indents* 0)
(declaim (type boolean *verbose* *single-node*)
	 (type fixnum *backward-indents*))

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

(defparameter *lazy-backwards* nil)

(defmacro with-verbose (&body body)
  "In the codes below, the computation nodes will be displayed when (backward out)"
  `(let ((*verbose* t))
     ,@body))

(defmacro with-backend (backend &body body)
  "Switches a backend.

See also: define-node-extension"
  (declare (type keyword backend))
  `(let ((*default-backend* ,backend))
     ,@body))

(deftype TensorData ()
  `(or fixnum ; scalar values
       float
       null
       cons
       function

       simple-array
       mgl-mat:mat ; to be deleted
       ))

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

(defun waffe-array-p (c)
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

(defstruct Grad-Tmp
  (value nil)
  (grad-shape nil :type list)
  (grad-called nil :type boolean))

(defstruct (WaffeNodeThread
	    (:print-function
	     (lambda (obj stream depth)
		 (declare (ignore depth))
		 (format stream "[WaffeNodeThreadInfomation]:~%The top of node is: ~a~% The tensor locates in the depth of ~a~% The tensor is registered as a ~a~%"
			 (waffenodethread-belong-to obj)
			 (waffenodethread-thread-idx obj)
			 (waffenodethread-cache-n obj))))
	    (:constructor
		thread
		(thread-idx
		 belong-to
		 &aux
		   (thread-idx thread-idx)
		   (belong-to belong-to))))
  (belong-to nil :type (or null symbol))
  (thread-idx 0 :type fixnum)
  (cache-n 0 :type fixnum))

; To add: create tensors with (tensor a :debug t) which enables displaying more informations (e.g.: mat's memory-address)
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
				     (breakme? nil)
			     &aux (data (init-waffe-tensor-data value))
			       (backend (check-backend backend extend))
			       (grad nil)
			       (thread-data thread-data)
			       (destructive? t)
			       (is-next-destruct? breakme?)
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
  ; どの構造体が使われていでどれが使われていないかを書き直す
  (data nil :type Tensordata)
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

  (visible-area `(t) :type list)

  ; 下は怪しい
  (destructive? nil :type boolean)
  (thread-data nil :type (or waffenodethread null))
  (is-sysconst? nil :type boolean)
  (path-through-node? nil :type boolean)
  (tensor-ident nil :type (or null symbol))
  (force-ignore-jit nil :type boolean)
  (key nil :type (or null cons))
  (idx nil :type (or null symbol))
  (is-data-destructed? nil :type boolean))

(declaim (ftype (function (waffetensor) waffetensor) maybe-copy))
(defun maybe-copy (tensor)
  "Returns a tensor, if tensor is allowed to destruct. otherwise returns a copy of tensor"
  (declare (optimize (speed 3)))
  (if (waffetensor-is-next-destruct? tensor)
      tensor
      (sysconst (copy-mat (data tensor)))))

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
	 (:lazy-eval      (waffetensor-data tensor))
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

(defun reset-config ()
  "Set *no=grad*=nil, *in-node-method*=nil. This function is for the case when an error occurs during training and these parameters weren't reseted correctly."
  (setf *no-grad* nil)
  (setf *in-node-method* nil))

(defun (setf data) (val &optional tensor)
  "Modifying tensor'data.

When the argument that you want to insert is a tensor, this function automatically reads tensor's data and modify with it"
  (if (typep val 'waffetensor)
      (setf (waffetensor-data tensor) (data val))
      (setf (waffetensor-data tensor) val)))

(defun init-waffe-tensor-data (content)
  ; todo: coerce: simple-array -> mgl-mat
  
  (when (eql content t)
    (reset-config)
    (backward-error "An tensor is initialized with ~a, which means your node/model's parameter weren't updated. This is due to you're training with *no-grad*=t or cl-waffe::*in-node-method*=t. Try this after calling (cl-waffe:reset-config). (This is occurs when an error occurs during training models.)" content))
  
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

(defmacro !allow-destruct (tensor)
  "Tensors which path through this macro are allowed to be destructed by cl-waffe's kernel.


In default, cl-waffe's operators won't make side effects.
@begin[lang=lisp](code)
(setq a (!randn `(3 3)))

;#Const(((0.811... -0.43... -0.91...)        
;                 ...
;        (0.959... -0.62... 1.150...)) :mgl t :shape (3 3))

(!exp a)
;#Const(((2.252... 0.645... 0.400...)        
;                 ...
;        (2.610... 0.534... 3.159...)) :mgl t :shape (3 3))

(print a)
;#Const(((0.811... -0.43... -0.91...)        
;                 ...
;        (0.959... -0.62... 1.150...)) :mgl t :shape (3 3))
@end[lang=lisp](code)

However, This macro let kernel know that the given tensor is allowed to destruct(i.e.: the result is overwritten)

@begin[lang=lisp](code)
(setq a (!randn `(3 3)))

;#Const(((0.811... -0.43... -0.91...)        
;                 ...
;        (0.959... -0.62... 1.150...)) :mgl t :shape (3 3))

(!allow-destruct a)
; T

(!exp a)
;#Const(((2.252... 0.645... 0.400...)        
;                 ...
;        (2.610... 0.534... 3.159...)) :mgl t :shape (3 3))

(print a) ; You can see the result is overwritten.
;#Const(((2.252... 0.645... 0.400...)        
;                 ...
;        (2.610... 0.534... 3.159...)) :mgl t :shape (3 3))
@end[lang=lisp](code)

Avoiding copy, destructive operations are superior in terms of memory usage.

@begin[lang=lisp](code)
(setq a (!randn `(100 100)))

(time (!exp a))
;Evaluation took:
;  0.000 seconds of real time
;  0.000275 seconds of total run time (0.000219 user, 0.000056 system)
;  100.00% CPU
;  498,150 processor cycles
;  31,264 bytes consed

(!allow-destruct a)

(time (!exp a))
; Evaluation took:
;  0.000 seconds of real time
;  0.000178 seconds of total run time (0.000160 user, 0.000018 system)
;  100.00% CPU
;  273,646 processor cycles
;  0 bytes consed 
@end[lang=lisp](code)

See also: !disallow-destruct which does the opposite.
"
  `(setf (waffetensor-is-next-destruct? ,tensor) t))

(defmacro !disallow-destruct (tensor)
  "Tensors that path through this macro are not destructed.

@begin[lang=lisp](code)
(setq a (!randn `(3 3)))
;#Const(((1.084... -1.10... 1.406...)        
;                 ...
;        (1.044... 0.059... -0.53...)) :mgl t :shape (3 3))

(!allow-destruct a)
; T
(!disallow-destruct a)
; NIL

(!exp a)
;#Const(((2.957... 0.329... 4.080...)        
;                 ...
;        (2.840... 1.060... 0.584...)) :mgl t :shape (3 3))

(print a) ; a is kept remained.
;#Const(((1.084... -1.10... 1.406...)        
;                 ...
;        (1.044... 0.059... -0.53...)) :mgl t :shape (3 3))
@end[lang=lisp](code)"
  `(setf (waffetensor-is-next-destruct? ,tensor) nil))

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
       (backward-error "Grads are only created for waffetensor. But got ~a" ,tensor))
     
     (unless (waffetensor-grad ,tensor)
       (backward-error "The tensor is not a parameter. Constants doesn't have a grad. If you need grad, please define it with (parameter (const XXX)). When using ~a~%" ,tensor))

     (if (typep (waffetensor-grad ,tensor) 'cons)
	 (backward-error "Refering the tensor's grad, cl-waffe got nil.~%This is because (backward out) weren't called after using (zero-grad), otherwise the computation nodes aren't continuous. ~%See also: Documentation of defnode/defmodel. ~%When using ~%~a~%" ,tensor))
	 
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
  (declare (optimize (speed 3))
	   (type waffetensor tensor))
  (if (typep (data tensor) 'mgl-mat:mat)
      (unless (eq (!shape tensor) `(1))
	(backward-error "grad can be implicitly created only for scalar outputs")))
  (let ((*no-grad* t))
    (let ((*lazy-backwards* (make-hash-table)))
      (labels ((backward-by-id (id lazy-tensors)
		 (remhash id *lazy-backwards*)
		 ; creates grad-tmp tensor

		 (let* ((first-tensor (cdr (car lazy-tensors)))
			(result-shape (grad-tmp-grad-shape
				       (waffetensor-grad-tmp first-tensor)))
			(result-tmp (!zeros result-shape)))
		   (dolist (lazy-tensor-info lazy-tensors)
		     (apply
		      #'!write-faref
		      result-tmp
		      (cdr lazy-tensor-info)
		      (car lazy-tensor-info)))

		   (setf (waffetensor-backward result-tmp)
			 (waffetensor-backward first-tensor))
		   (setf (waffetensor-state result-tmp)
			 (waffetensor-state first-tensor))
		   (setf (waffetensor-variables result-tmp)
			 (waffetensor-variables first-tensor))
		   (setf (waffetensor-is-ancestor-param result-tmp) t)

		   (when *verbose*
		     (format t "Resumption from Lazy Evaluated==~%"))
		   (setfgradtmp result-tmp result-tmp)
		   (backward1 result-tmp))))
	(backward1 tensor)
	(loop while (not (= 0 (hash-table-count *lazy-backwards*)))
	      do (maphash #'backward-by-id *lazy-backwards*)))
	nil)))

(declaim (inline step-next-node))
(defun step-next-node (tensor n)
  (if (waffetensor-is-ancestor-param (nth-var tensor n))
      (backward1 (nth-var tensor n))))

(declaim (ftype (function (waffetensor) (values null &optional)) backward1))
(define-with-typevar backward1 u (tensor)
  (declare (optimize (speed 3) (safety 0))
	   (type waffetensor tensor))
  
  ; Displaying Backward Nodes, when *verbose* is t.
  (when *verbose*
    (dotimes (_ (the fixnum *backward-indents*))
      (format t " "))
    (format t "~a~%" (if (null (waffetensor-state tensor))
			 "<The End of Node>"
			 (waffetensor-state tensor))))
  
  (cond
    ((waffetensor-backward tensor) ;Backward exists?
     (let* ((grad-tmp-before (waffetensor-grad-tmp tensor))
	    (grad-before (if (grad-tmp-grad-called grad-tmp-before) ;check if the node is a top
			     (grad-tmp-value grad-tmp-before)
			     (sysconst 1.0))))
       ; calculating backward(state, dy) -> x.grad, y.grad...

       ; Update Parameters
       (let ((*single-node* (= 1 (length (the list (waffetensor-variables tensor)))))
	     (*backward-indents* (if *verbose*
				     (if *single-node*
					 (+ *backward-indents* 1)
					 *backward-indents*)
				     0)))
	 ; Each node has its own specific optimisation described below.
	 (cond
	   ((and
	     (waffetensor-state
	      (car
	       (waffetensor-variables tensor)))
	     (areftensor-p
	      (waffetensor-state tensor)))
	    #| Explain: When Node is Like...
	    [Node: ArefTensor {A1}]
	      [Node: AddTensor {0}]|
	    [Node: ArefTensor {A2}]
	      [Node: AddTensor {0}]
	    Stops exploring deeper of addtensor until all areftensor will be registered.
	    |#
	    (let* ((grad-before (grad-tmp-value grad-tmp-before))
		   (variables (waffetensor-variables tensor))
		   (state (waffetensor-state tensor))
		   (called-from-state (waffetensor-state (car variables)))
		   (higher-node-id (slot-value called-from-state 'model-ident)))
	      (unless (= 1 (length variables))
		(backward-error "cl-waffe's internal error: the size of !aref's retent should be 1 but got: ~a" variables))

	      ; Pseudo, moves down one node.
	      (setf (waffetensor-backward grad-before)
		    (waffetensor-backward (car variables)))
	      (setf (waffetensor-state grad-before)
		    (waffetensor-state (car variables)))
	      (setf (waffetensor-variables grad-before)
		    (waffetensor-variables (car variables)))
	      (setf (waffetensor-is-ancestor-param grad-before) t)
	      (setf (grad-tmp-grad-shape
		     (waffetensor-grad-tmp grad-before))
		    (!shape (car variables)))
	      (push (cons
		     (areftensor-shape state)
		     grad-before)
		    (gethash
		     higher-node-id
		     *lazy-backwards*)))
	    (when *verbose*
	      (dotimes (_ (the fixnum *backward-indents*))
		(format t " "))
	      (format t "<Lazy Evaluated>~%"))
	    nil)
	   ; Otherwise, simply explore deeper nodes if there's param.
	   (T 
	    (let ((grads (call-backward
			  (waffetensor-state tensor)
			  grad-before)))
	      (declare (type list grads)) ; Todo: Print Error
	      (unless (= (length (waffetensor-variables tensor))
			 (length grads))
		(backward-error "Backward Error: The number of arguments :forward has doesnt correspond with of :backward returned in list.~%Node: ~a~%Forward:~a~%:Backward:~a~%" (waffetensor-state tensor) (waffetensor-variables tensor) grads))

	      (dotimes (n (length grads))
		(unless (eql (data (nth n grads)) nil) ; when nil, ignored.
		  (setf (waffetensor-thread-data (nth n grads))
			(waffetensor-thread-data tensor))
		  (setfgradtmp (nth-var tensor n) (nth n grads))
		  (step-next-node tensor n))))
	    nil)))))
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
		       (+ (the u (waffetensor-grad tensor))
			  (the u
			       (value (grad-tmp-value
				 (waffetensor-grad-tmp tensor))))))))))))
  nil)

; Initializer of tensors

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

(defun !view ()
  "Todo:
Its usage is similar to pytorch's view.
!view returns a displaced and reshaped tensor but won't produce copies.
This is useful when you want to apply some operations to the specified area of tensor. !set-batch will be implemented by view.
Excepted Usage.
(with-view tensor '(0 3) t t
-> x
do (exp x))...")

(declaim (ftype (function (waffetensor &rest t) waffetensor) !aref))
(defun !aref (tensor &rest dims)
  "!aref creates a new tensor from the area specified by @cl:param(dims) from the given @cl:param(tensor).

This function is setfable and both function produces the computation nodes.


dims is consisted of list, and each dimension is described as follow formats:

@begin(deflist)
@def(t)
@term(t means (0~max-len) in the dimension.)
@def(fixnum)
@term(copies the index of fixnum in the dimension.)
@def(list)
@term(list must be of (start stop), copying tensors from start to stop in the dimension. that is, the result in the dimension is the copy of: @b(start<=x<stop).
Using t as @cl:param(stop) means: t is the last element in the dimension.)
@end(deflist)

The fixnum used in @cl:param(dims) is not only positive numbers but also negative numbers.

For example, -1 is interpreted as (+ maxlen -1), -2 is interpreted as (+ maxlen -2)...

Note: (setf !aref) overwrites the given tensor's mat but won't overwrites its computation node. in order to update nodes, you must write it like: (setq a (setf (!aref a ...) ...))... See Example for the details.

Tensor cut-outs act on:
@begin(deflist)
@def(When is not setf)
@term(act on the given tensor.)
@def(When is setf)
@term(act on the target tensor. (e.g.: (setf (!aref target-tensor ...) input-tensor)))
@end(deflist)

Example:
@begin[lang=lisp](code)
(setq a (!randn `(10 5 3)))
;#Const((((0.621... -1.15... 2.396...)         
;                   ...
;         (0.157... 0.389... 1.084...))        
;                 ...
;        ((1.123... -0.58... -0.28...)         
;                   ...
;         (0.506... -0.44... -0.26...))) :mgl t :shape (10 5 3))

(!aref a '(0 3)) ; interpreted as (!aref a '(0 3) t t)
;#Const((((0.621... -1.15... 2.396...)         
;                   ...
;         (0.157... 0.389... 1.084...))        
;                 ...
;        ((0.694... 0.954... 1.210...)        
;                   ...
;         (0.884... 0.059... 0.190...))) :mgl t :shape (3 5 3))

(!aref a '(1 3))
;#Const((((0.657... 0.834... -2.01...)         
;                   ...
;         (1.194... 0.517... 0.356...))
;        ((0.694... 0.954... 1.210...)         
;                   ...
;         (0.884... 0.059... 0.190...))) :mgl t :shape (2 5 3))

(!aref a '(1 0)) ; When (cdr dims) <= 0, interpreted as (- (!shape tensor dim) (cdr dims))
; In this Example, this is the same as (!aref a '(1 10))
;#Const((((0.657... 0.834... -2.01...)         
;                   ...
;         (1.194... 0.517... 0.356...))        
;                 ...
;        ((1.123... -0.58... -0.28...)         
;                   ...
;         (0.506... -0.44... -0.26...))) :mgl t :shape (9 5 3))

(!aref a '(1 -1))
;#Const((((0.657... 0.834... -2.01...)         
;                   ...
;         (1.194... 0.517... 0.356...))        
;                 ...
;        ((-2.29... -1.12... -0.68...)         
;                   ...
;         (-1.74... 0.489... 1.519...))) :mgl t :shape (8 5 3))

(!aref a t '(0 2))
;Tensors in lower dimensions can also be clipped.
;If 0th dim isn't needed to be cut, place t.
;#Const((((0.621... -1.15... 2.396...)
;         (0.642... 0.029... 1.334...))        
;                 ...
;        ((1.123... -0.58... -0.28...)
;         (-2.43... -0.29... 0.882...))) :mgl t :shape (10 2 3))

(!aref a '(0 2) '(1 2) '(1 3))
;#Const((((0.029... 1.334...))
;        ((-1.41... -0.32...))) :mgl t :shape (2 1 2))

; This function is setfable, but currently I won't come up with the best solution to update computation node.
; I know it is very ugly but additional setq is required after setf.
; Also, note that (setf !aref). overwrites a.
(setq a (setf (!aref a '(0 3) '(0 3)) (!zeros '(3 3))))

;#Const((((0.0 0.0 0.0)         
;                   ...
;         (0.157... 0.389... 1.084...))
;                 ...
;        ((1.123... -0.58... -0.28...)
;                   ...
;         (0.506... -0.44... -0.26...))) :mgl t :shape (10 5 3))

(!aref a 0 0)
;#Const((((0.0 0.0 0.0))) :mgl t :shape (1 1 3))
@end[lang=lisp](code)"
  (call (ArefTensor dims) tensor))

(defun !areflist (tensor dims)
  (call (ArefTensor dims) tensor))

(defun (setf !aref) (value tensor &rest dims)
  "Todo: Define it as macro and (setq tensor ~)"
  (multiple-value-bind (value tensor) (straighten-up (assure-tensor value) (assure-tensor tensor))
    (let ((result (call (SetfArefTensor dims)
			tensor
			value)))
      ; (setq tensor (setf (!aref ..)) will destruct model-ident so update it.
      (when (and
	     (waffetensor-state result)
	     (waffetensor-state tensor))
	(setf (slot-value
	       (waffetensor-state result)
	       'model-ident)
	      (slot-value
	       (waffetensor-state tensor)
	       'model-ident)))
      result)))

(defun (setf !areflist) (value tensor dims)
  ; For backward, you need to call it like (setq z (setf (!aref x ~) ~))
  ; To solve this problem, i guess i need more macros.
  (multiple-value-bind (value tensor) (straighten-up (assure-tensor value) (assure-tensor tensor))
    (call (SetfArefTensor dims)
	  tensor
	  value)))

(defmacro !row-major-aref (tensor index)
  `(mgl-mat:row-major-mref (data ,tensor) ,index))

(defmacro !with-mgl-operation (tensor var &body body)
  `(let ((,var (data ,tensor)))
     ,@body))

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

(define-with-typevar !where u (condition tensor then else)
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
      (declare (type (simple-array u) result-array tensor-array))
      (loop for i fixnum upfrom 0 below (!size tensor)
	    do (setf (aref result-array i)
		     (if (funcall condition (aref tensor-array i))
			 then
			 else))))
    result))

(defun !index () "Todo")

(define-with-typevar !filter u (tensor lambda)
  "Applying every tensor's element @cl:param(lambda), it returns an tensor which comprised of the @cl:param(lambda)'s returned values.

@begin(deflist)
@def(tensor)
@term(an tensor that to be refered to)
@def(lambda)
@term(an function that returns elements at position @cl:param(x))
@end(deflist)
@begin[lang=lisp](code)
(setq tensor (!randn `(10 10)))
(!filter tensor #'(lambda (x) (if (> x 0) x 1.0)))
;#Const(((0.802... 1.331... ~ 0.998... 1.994...)        
;                 ...
;        (1.0 0.005... ~ 0.296... 0.358...)) :mgl t :shape (10 10))
@end[lang=lisp](code)"

  (declare (optimize (speed 3))
	   (type function lambda)
	   (type waffetensor tensor))
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
      (declare (type (simple-array u) result-array tensor-array))
      (loop for i fixnum upfrom 0 below (!size tensor)
	    do (setf (aref result-array i)
		     (funcall lambda (aref tensor-array i))))
    result)))

(defun write-description (res backward backend)
  (declare (ignore backward res backend))
  ; Parameter { ... <= here }
  ;(write-string (format nil " :device :~a" backend) res)
  )

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
      (error "internal bug"))
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

      (typecase contents
	(mgl-mat:mat
	 (pprint-vector res (mgl-mat:mat-to-array contents) newline
			(if (null grad)
			    (+ indent-size (length "#Const("))
			    (+ indent-size (length "#Parameter{"))))
	 (write-string
	  (format nil " :dtype :~(~a~) :shape ~a :backward ~a"
		  (mat-ctype contents)
		  (mgl-mat:mat-dimensions contents)
		  (waffetensor-state tensor))
	  res)
	 (unless (null grad)
	   (write-description res backward backend))
	 (if (null grad)
	     (write-string ")" res)
	     (write-string "}" res)))
	(function
	 (cond
	   ((lazy-transpose-p tensor)
	    (format res "<Transposed Tensor> :shape ~a :backward ~a"
		    (!shape tensor)
		    (waffetensor-state tensor)))
	   (T
	    (format res "<LazyEvaluatedTensor> :shape ~a :backward ~a"
		    (!shape tensor)
		    (waffetensor-state tensor))))
	 (if (null grad)
	     (write-string ")" res)
	     (write-string "}" res)))
	(T
	 (write-string (format nil "~A" contents) res)
	 (write-string
	  (format nil " :dtype ~a :backward ~a"
		  (type-of contents)
		  (waffetensor-state tensor))
	  res)
	 (unless (null grad)
	   (write-description res backward backend))
	 (if (null grad)
	     (write-string ")" res)
	     (write-string "}" res))))
      res)))

