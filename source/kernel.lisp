
(in-package :cl-waffe)

(defparameter *kernels* `(:cpu :opencl))
(defparameter *instructions* `(:add
			       :sub
			       :mul
			       :div
			       :log
			       :pow
			       :sum
			       :mean
			       :dot
			       :exp
			       :tanh
			       :reshape
			       :transpose
			       :repeat))

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

(defun callop (instruction &rest variables)
  (unless (find instruction *instructions*) ;doesnt works?
    (error "unsupported instruction: ~a" instruction))

  (unless (assure-tensors variables)
    (error "all inputs must have same backends and be waffe tensor"))

  (let* ((backend (slot-value (first variables) 'backend))
	 (args (map 'list (lambda (x) (let ((c (slot-value x 'data))) ; numcl check, ituka kesu
					(if (or (typep c 'array) (typep c 'vector))
					    (numcl:asarray c)
					    c)))
		    variables))
	 (result (case backend
		   (:cpu (cl-waffe.backends.cpu:kernel instruction args))
		   (:opencl (cl-waffe.backends.opencl:kernel instruction args)))))
    (const result :backend backend)))

(defun backends-available ())

(defun check-supported-instruction (backend))

