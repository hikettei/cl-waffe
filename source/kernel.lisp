
(in-package :cl-waffe)

; dispaches kernel based on backends. and optimize node

(defparameter *kernels* `(:mgl)
  "The list of cl-waffe supported kernels")

(defparameter *destructive-operation* nil
  "When t, some computations will be done destructively. Default is nil")

(defun share-memory-p (tensor1 tensor2)
  "Todo"
  (declare (ignore tensor1 tensor2)))

(defmacro with-optimized-operation (&body body)
  ; doing all operations with destructive
  `(progn
     (setf *destructive-operation* t)
     (let ((result (progn ,@body)))
       (setf *destructive-operation* nil)
       result)))

(defmacro with-jit (&body body)
  `(progn
     (setf cl-waffe.backends.mgl::*force-disable-jit* nil)
     (setf cl-waffe.backends.mgl:*force-lazy-eval* t)
     (setf cl-waffe.backends.mgl:*verbose* t)
     ,@body))

(defmacro with-no-jit (&body body)
  `(progn
     (setf cl-waffe.backends.mgl::*force-disable-jit* t)
     (setf cl-waffe.backends.mgl:*force-lazy-eval* nil)
     (setf cl-waffe.backends.mgl:*verbose* t)
     ,@body))

(declaim (ftype (function (waffetensor) waffetensor) warranty))
(defun warranty (tensor)
  "Notice waffe's optimizer that do not delete tensor given until warranty called
   in the calc node.

 When you encountered error in the forward step that the tensor that attempted to read has already cached ~, try this like:

@begin[lang=lisp](code)
(warranty your-tensor)
(print your-tensor)
@end[lang=lisp](code)"
  (declare (optimize (speed 3) (safety 0) (space 0))
	   (type waffetensor tensor))
  (prog1
      tensor
    (let ((thread (waffetensor-thread-data tensor)))
      (if thread (incf (waffenodethread-cache-n thread) 1)))))

#|
(declaim (ftype (function (waffetensor)
			  (or mat
			      simple-array
			      single-float
			      fixnum
			      function))
		value))|#
(defun value (tensor &key (ignore-transpose nil))
  "Access tensor's data, but if tensor is lazy-evaluated, eval them.

Note: this is not setfable"
  (declare (optimize (speed 3))
	   (type waffetensor tensor))
  
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
       (if (and (eql function-info :lazy-transpose) ignore-transpose)
	   ; do not step.
	   (setf (data tensor)
		 (the mgl-mat:mat
		      (funcall (the function (waffetensor-data tensor))
			       tensor nil nil)))
	   (setf (data tensor)
		 (the mgl-mat:mat
		      (cl-waffe.backends.mgl:compile-and-run-lazy tensor))))))
    (T (setf (data tensor) (data tensor)))))

;(declaim (ftype (function (keyword cons) waffetensor) invoke-mgl-kernel invoke-cpu-kenel))
(defun invoke-mgl-kernel (kernel-function variables &key (output nil) (overwrite nil))
  (declare (optimize (speed 3)))
  (sysconst (cl-waffe.backends.mgl:dispatch-kernel
				  kernel-function
				  *destructive-operation*
				  (car variables)
				  (second variables)
				  variables
				  :output output
				  :overwrite overwrite)
	    :thread-data (let ((r (find t variables
					:test (lambda (x y)
						(declare (ignore x))
						(waffetensor-thread-data y)))))
			   (if r
			       (waffetensor-thread-data r)
			       nil))
	    :path-through-node? (find t (map 'list
					     #'waffetensor-path-through-node?
					     variables))))

(defun invoke-cpu-kernel (kernel-function variables)
  (declare (optimize (speed 3)))
  (sysconst (cl-waffe.backends.cpu:dispatch-kernel kernel-function variables)
	    :thread-data (let ((r (find t (the list variables)
					:test (lambda (x y)
						(declare (ignore x))
						(waffetensor-thread-data y)))))
			   (if r
			       (waffetensor-thread-data r)
			       nil))
	    :path-through-node? (find t (map 'list
					     #'waffetensor-path-through-node?
					     variables))))

(defun invoke-kernel (kernel-function
		      variables
		      first-argument
		      i
		      &key
			(output nil)
			(overwrite nil))
  (declare (optimize (speed 3))
	   (type boolean overwrite)
	   (ignore first-argument i))
  (let ((has-mat (member-if #'(lambda (x)
				(when (typep x 'waffetensor)
				  (or (typep (data x) 'mgl-mat:mat)
				      (typep (data x) 'function))))
			    variables)))
    (if (null has-mat) ; invoke cpu-kernel
	(invoke-cpu-kernel kernel-function
			   variables)
	(invoke-mgl-kernel kernel-function
			   variables
			   :output output
			   :overwrite overwrite))))

(defun call-and-dispatch-kernel (kernel-function output overwrite &rest args)
  "Invoke kernel and run kernel-function. return new sysconst
It's the most general way for users to access cl-waffe's kernel.

If output is specified, write a result to output destructively.

If overwrite is t, side effect will occurs. (i.e.: args can be destructed)"
  (declare (inline invoke-kernel))
  (invoke-kernel kernel-function args (data (car args)) 0
		 :output output
		 :overwrite overwrite))

(declaim (ftype (function (keyword &rest waffetensor) waffetensor)))
(defun with-searching-calc-node (kernel-function &rest args)
  "Invoke kernel and run kernel-function. return new sysconst.

Todo:More Details"
  (declare (optimize (speed 3) (space 0) (safety 0))
	   (type keyword kernel-function)
	   (inline invoke-kernel))
  (invoke-kernel kernel-function args (data (car args)) 0))

(defgeneric with-searching-calc-node-optim (kernel-function target-data target-tensor args))

(defmethod with-searching-calc-node-optim (kernel-function (target-data mgl-mat:mat) target-tensor args)
  (declare (optimize (speed 3) (space 0) (safety 0))
	   (type keyword kernel-function))
  (invoke-kernel kernel-function `(,target-tensor ,@args) target-data 0)
  target-tensor)

(defmethod with-searching-calc-node-optim (kernel-function (target-data function) target-tensor args)
  (declare (optimize (speed 3) (space 0) (safety 0))
	   (type keyword kernel-function))
  (invoke-kernel kernel-function `(,target-tensor ,@args) target-data 0)
  target-tensor)

(defmethod with-searching-calc-node-optim (kernel-function target-data target-tensor args)
  (declare (optimize (speed 3) (space 0) (safety 0))
	   (type keyword kernel-function))
  (setf (data target-tensor) (data (invoke-kernel kernel-function `(,target-tensor ,@args) target-data 0)))
  target-tensor)

(defmacro with-kernel-case (target var &key (mgl nil) (mgl-cuda nil) (copy t) &aux (out (gensym)))
  "Reading the target's device, this macro invokes property codes described in :mgl, :mgl-cuda etc...

   Dynamically defining and caching cpu and cuda kernel.

   Every time reaches this macro, cl-waffe caches the target (i.e. the target is allowed to be destructed).

   This macro won't create computation nodes.

   The available slot is in *kernels*

   When :mgl-cuda is nil, automatically calls :mgl

   This macro returns the last value of called slots.

   The last value of :mgl, :mgl-cuda and so on, must be type of list (cons), or mgl-mat:mat, waffetensorcontenttype.

   Note: the target's thread-data must be already created. (i.e. By the time tensors reach this macro, at least once they needed to be pathed through Trainer or Model.)
   So, use this macro when you defining :forward and :backward in defnode macro because in defnode, backprop is disabled and computation nodes isn't always required.

Inputs
@begin(deflist)
@term(target)
@def(an target tensor.)
@term(var)
@def(where an copied tensor of target will be assigned.)
@term(:mgl)
@def(mgl-mat, when using cpu.)
@term(:mgl-mat)
@def(mgl-mat, when using cuda.)
@end(deflist)

Return: An tensor (where tensor is made by sysconst)

Example:

@begin[lang=lisp](code)
 (with-kernel-case x o
     :mgl (progn
            (axpy! 1.0 a o)) ; axpy! = !add
     :mgl-cuda nil) => #Const(((0.0 1.0 ~ 2.0 3.0)        
                 ...
      (0.0 4.0 ~ 5.0 6.0)) :mgl t :shape (10 10))

 ; This is useful when defining :backward
 (with-kernel-case x o
     :mgl (progn
            (list 1 1)))
@end[lang=lisp](code)"
  `(progn
     (unless (typep ,target 'waffetensor)
       (error "cl-waffe.with-kernel-case: target must be waffetensor. Encounted type of ~a, when using ~a" (type-of ,target) ,target))
     (value ,target)
     (cl-waffe.caches:with-cache
	 (,var ,target
	  :copy ,copy)
       (warranty ,target)
       (labels ((mgl-cpu-step  () ,@mgl)
		(mgl-cuda-step ()
		  (if (null ',mgl-cuda)
		      (mgl-cpu-step)
		      ,@mgl-cuda)))
	 (let ((,out (case (waffetensor-backend ,target)
		      (:mgl (if (use-cuda-p (data ,target))
					    (mgl-cuda-step)
					    (mgl-cpu-step)))
		      (T (error "cl-waffe.with-kernel-case: Encounted unsupported kernel ~a, cl-waffe supports kernel following: ~a"
				(waffetensor-backend ,target)
				*kernels*)))))
	   (typecase ,out
	     (list
	      (map 'list (lambda (x)
			   (sysconst x :thread-data ,target
		  		       :path-through-node? (waffetensor-path-through-node? ,target)))
		   ,out))
	     (T
	      (sysconst
	       ,out
	       :thread-data
	       (waffetensor-thread-data ,target)
	       :path-through-node? (waffetensor-path-through-node? ,target)))))))))
