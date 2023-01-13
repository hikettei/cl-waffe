
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
  (length 0 :type fixnum)
  (sp 0 :type fixnum)
  (lock nil :type boolean)
  (report-identifier 0 :type fixnum)
  (destruct-positions (make-hash-table) :type hash-table))


(defun report-reg (report var is-first-call?)
  (declare (optimize (speed 3) (space 0) (safety 0))
	   (type networkvariablereport report)
	   (type waffetensor var)
	   (type boolean is-first-call?))
  (if (car (waffetensor-report-index var))
      (let* ((old-report (gethash (car (waffetensor-report-index var))
	   			  (networkvariablereport-destruct-positions report)))
	     (new-report (if is-first-call?
			     (+ old-report 1)
			     old-report)))
	(declare (type fixnum old-report))
	(setf (gethash  (car (waffetensor-report-index var))
			(networkvariablereport-destruct-positions report))
	       new-report)))
  (if is-first-call?
      (progn
	(setf (waffetensor-report-index var)
	      `(,(second (waffetensor-report-index var))
		,(networkvariablereport-length report)))
	(incf (networkvariablereport-length report) 1)
	(setf (gethash
	       (second (waffetensor-report-index var))
	       (networkvariablereport-destruct-positions report))
	      0))))

(defun refer-report (report variables)
  (declare (optimize (speed 3) (space 0) (safety 0))
           (type networkvariablereport report))
  (let ((ls (networkvariablereport-destruct-positions report)))
    (map 'list (lambda (v)
		 (declare (type waffetensor v))
		 (let* ((index (networkvariablereport-sp report))
			(rp (gethash index ls)))
		   (incf (networkvariablereport-sp report) 1)
		   (if (waffetensor-destructive? v)
		       (case rp
			 (0 v)
			 (T nil))
		       nil)))
	 variables)))

(defun find-report (variables)
  (unless *ignore-optimizer*
    (let ((tensor (find t variables :test #'(lambda (x y)
					      (waffetensor-optim-report y)))))
      (unless (null tensor)
	  (waffetensor-optim-report tensor)
	  nil))))

(declaim (dtype (function (keyword cons) waffetensor) invoke-mgl-kernel invoke-cpu-kenel))
(defun invoke-mgl-kernel (kernel-function variables)
  (sysconst (cl-waffe.backends.mgl:dispatch-kernel kernel-function t (car variables) (second variables) variables)))

(defun invoke-cpu-kernel (kernel-function variables)
  (sysconst (cl-waffe.backends.cpu:dispatch-kernel kernel-function variables)))

(defgeneric invoke-kernel (kernel-function variables first-argument i))

(defmethod invoke-kernel (kernel-function
			  (variables cons)
			  (first-argument mgl-mat:mat)
			  (i fixnum))
  (declare (optimize (speed 3) (space 0) (safety 0))
	   (ignore first-argument i))
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
  (declare (optimize (speed 3) (space 0) (space 0))
	   (type keyword kernel-function))
  (invoke-kernel kernel-function args (data (car args)) 0))

