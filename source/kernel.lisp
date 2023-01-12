
(in-package :cl-waffe)

(declaim (inline callop))

; dispaches kernel based on backends. and optimize node

(defparameter *kernels* `(:mgl))
(defparameter *instructions* `(:add
			       :sub
			       :mul
			       :div
			       :log
			       :inv
			       :pow
			       :sqrt
			       :sum
			       :mean
			       :dot
			       :<
			       :matmul
			       :exp
			       :tanh
			       :reshape
			       :transpose
			       :repeat))

(defparameter *num-reports* 0)
(defparameter *ignore-optimizer* nil)

(defmacro with-ignore-optimizer (&body body)
  ; doing all operations with destructive
  `(progn
     (setf *ignore-optimizer* t)
     ,@body
     (setf *ignore-optimizer* nil)))

(defun check-kernel (variable)
  (unless (typep variable 'WaffeTensor)
    (error "The inputs must be tensor got: ~a" variable))
  
  (unless (find (slot-value variable 'backend) *kernels*)
    (error "Invaild kernel: ~a" (slot-value variable 'backend))
    T))

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
	   (type booleean is-first-call?))
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
					      (declare (ignore y))
					      (waffetensor-optim-report y)))))
      (unless (null tensor)
	  (waffetensor-optim-report tensor)
	  nil))))

(defun with-searching-calc-node (kernel-function &rest args)
  (let  ((is-all-array? (find t (map 'list (lambda (x) (declare (type waffetensor x)) (waffetensor-is-mat x)) args)))
	 (res-tensor nil)
	 (is-first-time-call? nil)
	 (report nil)
	 (destructable-variables nil)
	 (result nil))
     (if is-all-array?
	 (let* ((report (find-report args))
		(is-first-time-call? (if report
					 (not (networkvariablereport-lock report))
					 t))
		(destructable-variables (if (and report (not is-first-time-call?))
					    (refer-report report args)
					    args))
		(result (cl-waffe.backends.mgl:dispatch-kernel
			 kernel-function
			 is-first-time-call?
			 (car destructable-variables)
			 (second destructable-variables)
			 args)))
	   (setq res-tensor (sysconst result)))
	 (let* ((result (apply #'cl-waffe.backends.cpu:dispatch-kernel kernel-function args)))
	   (setq res-tensor (sysconst result))))
    (map 'list (lambda (x) (declare (type waffetensor x)) (incf (waffetensor-calln x) 1)) args)
    
    (if (and is-all-array? (null report) (not *ignore-optimizer*))
      (let ((any-param (find t args :test (lambda (x y)
					    (declare (ignore x)
						     (type waffetensor y))
					    (waffetensor-is-param? y)))))
	(unless (null any-param)
	  (progn (setf (waffetensor-optim-report any-param)
		       (make-networkvariablereport :length 0
						   :sp 0
						   :lock nil
						   :report-identifier *num-reports*))
		 (incf *num-reports* 1)
		 (setq report (waffetensor-optim-report any-param))))))
     
    (if (and report (not *ignore-optimizer*))
	(dolist (v args)
	  (report-reg report v is-first-time-call?)))

    (if (and report (not *ignore-optimizer*))
	(setf (waffetensor-optim-report res-tensor) report))

    (if (and (car destructable-variables) (not *ignore-optimizer*))
	(setf (waffetensor-report-index res-tensor) ;dop(a,b)a=a
	      (waffetensor-report-index (car destructable-variables))))

     res-tensor))

