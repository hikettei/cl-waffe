
(in-package :cl-waffe)

(declaim (inline callop))

; dispaches kernel based on backends. and optimize node

(defparameter *kernels* `(:mgl)
  "The list of cl-waffe supported kernels")

(defparameter *destructive-operation* nil
  "When t, some computations will be done destructively. Default is nil")

(defmacro with-optimized-operation (&body body)
  ; doing all operations with destructive
  `(progn
     (setf *destructive-operation* t)
     (let ((result (progn ,@body)))
       (setf *destructive-operation* nil)
       result)))

(declaim (ftype (function (waffetensor) waffetensor) warranty))
(defun warranty (tensor)
  "Notice waffe's optimizer that do not delete tensor given until warranty called
   in the calc node"
  (declare (optimize (speed 3) (safety 0) (space 0))
	   (type waffetensor tensor))
  (prog1
      tensor
    (let ((thread (waffetensor-thread-data tensor)))
      (if thread (incf (waffenodethread-cache-n thread) 1)))))

(declaim (ftype (function (keyword cons) waffetensor) invoke-mgl-kernel invoke-cpu-kenel))
(defun invoke-mgl-kernel (kernel-function variables)
  (sysconst (cl-waffe.backends.mgl:dispatch-kernel
				  kernel-function
				  *destructive-operation*
				  (car variables)
				  (second variables)
				  variables)
	    :thread-data (let ((r (find t variables
					:test (lambda (x y)
						(declare (ignore x))
						(waffetensor-thread-data y)))))
			   (if r
			       (waffetensor-thread-data r)
			       nil))))

(defun invoke-cpu-kernel (kernel-function variables)
  (sysconst (cl-waffe.backends.cpu:dispatch-kernel kernel-function variables)
	    :thread-data (let ((r (find t variables
					:test (lambda (x y)
						(declare (ignore x))
						(waffetensor-thread-data y)))))
			   (if r
			       (waffetensor-thread-data r)
			       nil))))

(defgeneric invoke-kernel (kernel-function variables first-argument i))
(defmethod invoke-kernel (kernel-function
			  (variables cons)
			  (first-argument mgl-mat:mat)
			  (i fixnum))
  (declare (optimize (speed 3) (space 0) (safety 0))
	   (ignore i first-argument))
  (invoke-mgl-kernel kernel-function variables))

(defmethod invoke-kernel (kernel-function
			  (variables cons)
			  first-argument
			  (i fixnum))
  (declare (optimize (speed 3) (space 0) (safety 0))
	   (ignore first-argument))
  (if (= i 0)
      (invoke-kernel kernel-function variables (data (second variables)) (+ i 1))
      (invoke-cpu-kernel kernel-function variables)))

(declaim (ftype (function (keyword &rest waffetensor) waffetensor)))
(defun with-searching-calc-node (kernel-function &rest args)
  (declare (optimize (speed 3) (space 0) (safety 0))
	   (type keyword kernel-function))
  (invoke-kernel kernel-function args (data (car args)) 0))

(defgeneric with-searching-calc-node-optim (kernel-function target-data target-tensor args))

(defmethod with-searching-calc-node-optim (kernel-function (target-data mgl-mat:mat) target-tensor args)
  (declare (optimize (speed 3) (space 0) (safety 0))
	   (type keyword kernel-function))
  (invoke-kernel kernel-function `(,target-tensor ,@args) target-data 0)
  target-tensor)

(defmethod with-searching-calc-node-optim (kernel-function target-data target-tensor args)
  (declare (optimize (speed 3) (space 0) (safety 0))
	   (type keyword kernel-function))
  (setf (data target-tensor) (data (invoke-kernel kernel-function `(,target-tensor ,@args) target-data 0)))
  target-tensor)

(defmacro with-kernel-case (target var &key (mgl nil) (mgl-cuda nil))
  "Reading the target's device, this macro invokes property codes described in :mgl, :mgl-cuda etc...

   Every time reaches this macro, cl-waffe caches the target (i.e. the target is allowed to be destructed).

   This macro won't create computation nodes.

   The available slot is in *kernels*

   When :mgl-cuda is nil, automatically calls :mgl

   This macro returns the last value of called slots.

   The last value of :mgl, :mgl-cuda and so on, must be type of list (cons), or mgl-mat:mat, waffetensorcontenttype.

   Note: the target's thread-data must be already created. (i.e. By the time tensors reach this macro, at least once they needed to be pathed through Trainer or Model.)
   So, use this macro when you defining :forward and :backward in defnode macro because in defnode, backprop is disabled and computation nodes isn't always required.

   Inputs: target, an target tensor.
           var, where an copied tensor of target will be assigned.
           :mgl mgl-mat, when using cpu.
           :mgl-cuda mgl-mat, when using cuda.

   Return: An tensor (where tensor is made by sysconst)

   Example:
   (with-kernel-case x o
       :mgl (progn
              (axpy! 1.0 a o)) ; axpy! = !add
       :mgl-cuda nil) => #Const(((0.0 1.0 ~ 2.0 3.0)        
                 ...
        (0.0 4.0 ~ 5.0 6.0)) :mgl t :shape (10 10))

   ; This is useful when defining :backward
   (with-kernel-case x o
       :mgl (progn
              (list 1 1)))"
  `(progn
     (unless (typep ,target 'waffetensor)
       (error "cl-waffe.with-kernel-case: target must be waffetensor. Encounted type of ~a, when using ~a" (type-of ,target) ,target))
     (cl-waffe.caches:with-cache
	 (,var ,target :place (cl-waffe.backends.mgl:create-thread-idx
			       (waffetensor-thread-data ,target)))
       (warranty ,target)
       (labels ((mgl-cpu-step  (,var) ,@mgl)
		(mgl-cuda-step (,var)
		  (if (null `,mgl-cuda)
		      (progn ,@mgl)
		      (progn ,@mgl-cuda))))
	 (let ((out (case (waffetensor-backend ,target)
		      (:mgl (if (use-cuda-p (data ,target))
					    (mgl-cuda-step ,var)
					    (mgl-cpu-step ,var)))
		      (T (error "cl-waffe.with-kernel-case: Encounted unsupported kernel ~a, cl-waffe supports kernel following: ~a"
				(waffetensor-kernel ,target)
				*kernels*)))))
	   (typecase out
	     (list
	      (map 'list (lambda (x)
			   (sysconst x :thread-data ,target))
		   out))
	     (T
	      (sysconst
	       out
	       :thread-data
	       (waffetensor-thread-data ,target)))))))))
