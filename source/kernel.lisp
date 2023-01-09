
(in-package :cl-waffe)

(declaim (inline callop))

(defparameter *kernels* `(:opencl :mgl))
(defparameter *instructions* `(:add
			       :sub
			       :mul
			       :div
			       :log
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
  `(prog1
     (setf *ignore-optimizer* t)
     ,@body
     (setf *ignore-optimizer* nil)))

(defun check-kernel (variable)
  (unless (typep variable 'WaffeTensor)
    (error "The inputs must be tensor got: ~a" variable))
  
  (unless (find (slot-value variable 'backend) *kernels*)
    (error "Invaild kernel: ~a" (slot-value variable 'backend))
    T))

(defun assure-tensors (variables)
  (check-kernel (first variables))
  (or (endp variables)
      (let ((x (slot-value (first variables) 'backend)))
	(every (lambda (y)
		 (check-kernel y)
		 (equal x (slot-value y 'backend)))
	       (rest variables)))))

(defstruct NetworkVariableReport
  length
  sp
  lock
  report-identifier
  destruct-positions)

(defun report-reg (report var)
  (if (null (waffetensor-report-index var))
      (progn
	(setf (waffetensor-report-index var)
	      (networkvariablereport-length report))
	(incf (networkvariablereport-length report) 1)
	(push `(,(waffetensor-calln var)
		,(waffetensor-destructively-calln var)
		;,(waffetensor-data var)
		)
	      (networkvariablereport-destruct-positions report)))
      (if (< (waffetensor-report-index var) (networkvariablereport-length report))
	  (let* ((old-report (nth (waffetensor-report-index var) (networkvariablereport-destruct-positions report)))
		 (new-report `(,(waffetensor-calln var)
			       ,(waffetensor-destructively-calln var)
			       ;,(waffetensor-data var)
			       )))
	    (setf (nth (waffetensor-report-index var) (networkvariablereport-destruct-positions report)) new-report))
	  (progn
	    (error "Failed to refer optimizing report, this is possibly due to interrupted network. index: ~a size: ~a"
		   (waffetensor-report-index var)
		   (networkvariablereport-length report))))))

(defun callop (instruction &rest variables)
  ;(declare (optimize (speed 3) (space 0) (safety 0) (debug 0)))
  (unless (find instruction *instructions*) ;doesnt works?
    (error "unsupported instruction: ~a" instruction))

  (unless (assure-tensors variables)
    (error "all inputs must have same backends and be waffe tensor"))
  (let* ((rp-exists? (find t variables :test (lambda (x y)
					       (declare (ignore y))
					       (typep (waffetensor-optim-report y) 'NetworkVariableReport))))
	 (report (if rp-exists? (waffetensor-optim-report rp-exists?) nil))
	 (is-first-call? (if report (not (networkvariablereport-lock report)) t))
	 (destructives (map 'list (lambda (x) (waffetensor-destructive? x)) variables))
	 (out (if (and (= 1 (length variables))
		       (car destructives))
		  (car variables))) ; optimize df(x) like log(x) where x is going to be abandoned
         (out            (if *ignore-optimizer* nil out))
	 (report         (if *ignore-optimizer* nil report))
	 (is-first-call? (if *ignore-optimizer* t is-first-call?))
	 (backend (waffetensor-backend (first variables)))
	 (args (map 'list (lambda (x) (waffetensor-data x)) variables)) ; do not make copy...
	 (all-not-array (every (lambda (x) (typep x 'waffesupporteddatatype)) args))
	 (result (case backend
		   (:cpu    (cl-waffe.backends.cpu:kernel instruction args out))
		   ;(:opencl (cl-waffe.backends.opencl:kernel instruction args out))
		   (:mgl    (if all-not-array ; Use CPU When like Const(1) + Const(1)
			        (cl-waffe.backends.cpu:kernel instruction args out)
				(cl-waffe.backends.mgl:kernel instruction args out variables (not is-first-call?))))
		   (T (error "No such backends: ~a" backend))))
	 (res-tensor (sysconst result :backend backend)))
    
    (map 'list (lambda (x) (incf (waffetensor-calln x) 1)) variables)
    
    (if (and (null report) (not *ignore-optimizer*))
      (let ((any-param (find t variables :test (lambda (x y)
						 (declare (ignore x))
						 (waffetensor-is-param? y)))))
	(unless (null any-param)
	  (progn (setf (waffetensor-optim-report any-param)
		       (make-networkvariablereport :length 0
						   :sp 0
						   :lock nil
						   :report-identifier *num-reports*
						   :destruct-positions nil))
		 (incf *num-reports* 1)
		 (setq report (waffetensor-optim-report any-param))))))

    (if (and report (not *ignore-optimizer*))
	(dolist (v variables)
	  (report-reg report v)))

    (if (and report (not *ignore-optimizer*))
	(setf (waffetensor-optim-report res-tensor) report))
    res-tensor))

(defun backends-available ())

(defun check-supported-instruction (backend))

