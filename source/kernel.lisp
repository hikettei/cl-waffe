
(in-package :cl-waffe)

(declaim (inline callop))

(defparameter *kernels* `(:mgl))
(defparameter *instructions* `(:data
			       :add
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

(defmacro warranty (tensor)
  ;return tensor's data but until this macro called, it is guaranteed that the data is not destructed
  `(data (callop :data ,tensor)))

(defmacro apply-destruct (out)
  `(progn
     (setf (waffetensor-is-data-destructed? ,out) t)
     (setf (waffetensor-destructively-calln ,out) 1)))

(defmacro with-ignore-optimizer (&body body)
  ; doing all operations with destructive
  `(progn
     (setf *ignore-optimizer* t)
     (let ((result (progn ,@body)))
       (setf *ignore-optimizer* nil)
       (data result))))

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

(defun report-reg (report var is-first-call?)
  (if (car (waffetensor-report-index var))
      (let* ((old-report (gethash (car (waffetensor-report-index var))
	   			  (networkvariablereport-destruct-positions report)))
	     (new-report (if is-first-call?
			     (+ old-report 1)
			     old-report)))
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
  (let ((ls (networkvariablereport-destruct-positions report)))
    (map 'list (lambda (v)
		 (let* ((index (networkvariablereport-sp report))
			(rp (gethash index ls)))
		   (incf (networkvariablereport-sp report) 1)
		   (if (waffetensor-destructive? v)
		       (case rp
			 (0 v)
			 (T nil))
		       nil)))
	 variables)))

(defun decide-nth (instruction ls) ;rev
  (if (eq instruction :div)
      (second ls)
      (car ls)))

(defparameter *n-to-i* (make-hash-table))

(defun callop (instruction &rest variables)
  ;(declare (optimize (speed 3) (space 0) (safety 0) (debug 0)))
  (unless (find instruction *instructions*) ;doesnt work?
    (error "unsupported instruction: ~a" instruction))

  (unless (assure-tensors variables)
    (error "all inputs must have same backends and be waffe tensor"))

  (let* ((rp-exists? (find t variables :test (lambda (x y)
					       (declare (ignore y))
					       (typep (waffetensor-optim-report y) 'NetworkVariableReport))))
	 (report (if rp-exists? (waffetensor-optim-report rp-exists?) nil))
	 (is-first-call? (if report (not (networkvariablereport-lock report)) t))
	 (destructives (if (and report (not is-first-call?) (not *ignore-optimizer*))
			   (refer-report report variables)
			   variables))
	 (out (if destructives
		  (decide-nth instruction destructives)
		  (decide-nth instruction variables))) ; optimize df(x) like log(x) where x is going to be abandoned
	 (out            (if *ignore-optimizer* out out))
	 (report         (if *ignore-optimizer* nil report))
	 (is-first-call? (if *ignore-optimizer* nil is-first-call?))
	 (backend (waffetensor-backend (first variables)))
	 (args (map 'list (lambda (x) (waffetensor-data x)) variables)) ; do not make so many copy...
	 (all-not-array (every (lambda (x) (typep x 'waffesupporteddatatype)) args))
	 (result (if (equal instruction :data)
		     (if out
			 (progn ;out = variables
			   (apply-destruct (car variables))
			   (if (and (not is-first-call?) (waffetensor-destructive? out))
			       (data out) ; apply destructive
			       (if (typep (data (car variables)) 'mgl-mat:mat)
				   (let ((o (mgl-mat:make-mat (mgl-mat:mat-dimensions (data (car variables))))))
				     (mgl-mat:copy! (data (car variables)) o)
				     o)
				   (data (car variables)))))
			 (if (typep (data (car variables)) 'mgl-mat:mat)
			     (let ((o (mgl-mat:make-mat (mgl-mat:mat-dimensions (data (car variables))))))
			       (mgl-mat:copy! (data (car variables)) o)
			       o)
			     (data (car variables))))
		     (case backend
		       (:cpu    (cl-waffe.backends.cpu:kernel instruction args out))				   
		       (:mgl    (if all-not-array ; Use CPU When like Const(1) + Const(1)
				    (cl-waffe.backends.cpu:kernel instruction args out)
				    (cl-waffe.backends.mgl:kernel instruction args out variables (not is-first-call?))))
		       (T (error "No such backends: ~a" backend)))))
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
						   :destruct-positions (make-hash-table)))
		 (incf *num-reports* 1)
		 (setq report (waffetensor-optim-report any-param))))))

    (if (and report (not *ignore-optimizer*))
	(dolist (v variables)
	  (report-reg report v is-first-call?)))

    (if (and report (not *ignore-optimizer*))
	(setf (waffetensor-optim-report res-tensor) report))

   (if out
	(setf (waffetensor-report-index res-tensor) ;dop(a,b)a=a
	      (waffetensor-report-index out)))

    res-tensor))

(defun backends-available ())

(defun check-supported-instruction (backend))

