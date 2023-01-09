
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

(defmacro apply-destruct (out tensor)
  `(progn
     (setf (waffetensor-report-index ,tensor)
	   (waffetensor-report-index ,out)) ;dop(a,b)a=a, wait for refresh
     (setf (waffetensor-is-data-destructed? ,tensor) t)
     (incf (waffetensor-destructively-calln ,tensor) 1)))

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
	(push `(0
		,(waffetensor-destructively-calln var))
	      (networkvariablereport-destruct-positions report)))
      (if (< (waffetensor-report-index var) (networkvariablereport-length report))
	  (let* ((old-report (nth (waffetensor-report-index var) (networkvariablereport-destruct-positions report)))
		 (new-report `(0
			       ,(waffetensor-destructively-calln var))))
	    (setf (nth (waffetensor-report-index var) (networkvariablereport-destruct-positions report)) new-report))
	  (progn
	    (error "Failed to refer optimizing report, this is possibly due to interrupted network. index: ~a size: ~a"
		   (waffetensor-report-index var)
		   (networkvariablereport-length report))))))

(defun incf-report (i report n var)
  (let* ((old-report (nth (waffetensor-report-index var) (networkvariablereport-destruct-positions report)))
	 (new-report `(,(+ (car old-report) n) ,(second old-report))))
    (setf (nth (waffetensor-report-index var) (networkvariablereport-destruct-positions report)) new-report)))

(defun refer-report (report variables)
  (let ((i (networkvariablereport-sp report))
	(ls (networkvariablereport-destruct-positions report)))
    (map 'list (lambda (v)
		 (let ((rp (nth i ls)))
		   (incf i 1)
		   (incf (networkvariablereport-sp report) 1)
		   (print ls)
		   (if (= (second rp) 0)
		       (if (waffetensor-destructive? v)
			   v
			   nil)
		       (if (and (= (mod (first rp) (second rp)) 0) ; 1+?
				(not (= (first rp) 0)))
			   (progn
			     (incf-report report i 1 v)
			     v)
			   (progn
			     (incf-report report i 1 v)
			     nil)))))
	 variables)))

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
	 (destructives (if (and report (not is-first-call?) (not *ignore-optimizer*)) (refer-report report variables) nil))
	 (out (if (and (= 1 (length variables))
		            (car destructives))
		  (car variables))) ; optimize df(x) like log(x) where x is going to be abandoned
         (out            (if *ignore-optimizer* nil out))
	 (report         (if *ignore-optimizer* nil report))
	 (is-first-call? (if *ignore-optimizer* t is-first-call?))
	 (backend (waffetensor-backend (first variables)))
	 (args (map 'list (lambda (x) (waffetensor-data x)) variables)) ; do not make copy...
	 (all-not-array (every (lambda (x) (typep x 'waffesupporteddatatype)) args))
	 (result (if (equal instruction :data)
		     (if *ignore-optimizer*
			 (if (typep (data (car variables)) 'mgl-mat:mat)
			     (mgl-mat:copy-mat (data (car variables)))
			     (data (car variables)))
			 (if out
			     (progn
			       (apply-destruct out (car variables))
			       (if (not is-first-call?)
				   (data out)
				   (if (typep (data (car variables)) 'mgl-mat:mat)
				       (let ((o (mgl-mat:make-mat (mgl-mat:mat-dimensions (data (car variables))))))
					 (mgl-mat:copy! (data (car variables)) o)
					 o)
				       (data (car variables)))))
			     (if (typep (data (car variables)) 'mgl-mat:mat)
				 (let ((o (mgl-mat:make-mat (mgl-mat:mat-dimensions (data (car variables))))))
				   (mgl-mat:copy! (data (car variables)) o)
				   o)
				 (data (car variables)))))
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
						   :destruct-positions nil))
		 (incf *num-reports* 1)
		 (setq report (waffetensor-optim-report any-param))))))

    (if (and report (not *ignore-optimizer*))
	(dolist (v variables)
	  (report-reg report v)))

    (if (and report (not *ignore-optimizer*))
	(setf (waffetensor-optim-report res-tensor) report))

    (print instruction)
    (print report)
    (print res-tensor)
    (print variables)
    (if (eq instruction :data)
	(print (data (car variables))))
    res-tensor))

(defun backends-available ())

(defun check-supported-instruction (backend))

