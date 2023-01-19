
(in-package :cl-waffe)

(declaim (inline callop))

; dispaches kernel based on backends. and optimize node

(defparameter *kernels* `(:mgl))

(defparameter *destructive-operation* nil)

(defmacro with-optimized-operation (&body body)
  ; doing all operations with destructive
  `(progn
     (setf *destructive-operation* t)
     (let ((result (prog1 ,@body)))
       (setf *destructive-operation* nil)
       result)))

(declaim (ftype (function (keyword cons) waffetensor) invoke-mgl-kernel invoke-cpu-kenel))
(defun invoke-mgl-kernel (kernel-function variables)
  (let ((result-tensor (sysconst (cl-waffe.backends.mgl:dispatch-kernel
				  kernel-function
				  *destructive-operation*
				  (car variables)
				  (second variables)
				  variables))))
    (if (or *no-grad* *destructive-operation*)
	; is result-tensor a copied mat?
	(if (and (not (waffetensor-is-data-destructed? (car variables)))
		 (if (second variables)
		     (not (waffetensor-is-data-destructed? (second variables)))
		     t))
	    (progn
	      ;(setf (waffetensor-is-data-destructed? result-tensor) t)
	      ;(!allow-destruct result-tensor)
	      result-tensor)
	    (progn ; destructed
	      (setf (waffetensor-is-next-destruct? result-tensor) nil)
	      result-tensor))
	result-tensor)))

(defun invoke-cpu-kernel (kernel-function variables)
  (sysconst (cl-waffe.backends.cpu:dispatch-kernel kernel-function variables)))

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

