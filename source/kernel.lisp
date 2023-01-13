
(in-package :cl-waffe)

(declaim (inline callop))

; dispaches kernel based on backends. and optimize node

(defparameter *kernels* `(:mgl))

(defparameter *num-reports* 0)
(defparameter *ignore-optimizer* t) ;nil

(defmacro with-ignore-optimizer (&body body)
  ; doing all operations with destructive
  `(progn
     (setf *ignore-optimizer* t)
     ,@body
     (setf *ignore-optimizer* nil)))

(defstruct NetworkVariableReport
  (length             0 :type fixnum)
  (sp                 0 :type fixnum)
  (lock               nil :type boolean)
  (report-identifier *num-reports* :type fixnum)
  (destruct-positions (make-hash-table) :type hash-table))


(declaim (dtype (function (keyword cons) waffetensor) invoke-mgl-kernel invoke-cpu-kenel))
(defun invoke-mgl-kernel (kernel-function variables tensor)
  (sysconst (cl-waffe.backends.mgl:dispatch-kernel kernel-function t (car variables) (second variables) variables)))

(defun invoke-cpu-kernel (kernel-function variables)
  (sysconst (cl-waffe.backends.cpu:dispatch-kernel kernel-function variables)))


(defgeneric invoke-kernel (kernel-function variables first-argument i))
(defmethod invoke-kernel (kernel-function
			  (variables cons)
			  (first-argument mgl-mat:mat)
			  (i fixnum))
  (declare (optimize (speed 3) (space 0) (safety 0))
	   (ignore i))
  (invoke-mgl-kernel kernel-function variables first-argument))

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
  (declare (optimize (speed 3) (space 0) (space 0))
	   (type keyword kernel-function))
  (invoke-kernel kernel-function args (data (car args)) 0))

