
(in-package :cl-waffe)

(defparameter *kernels* `(:cpu :opencl))
(defparameter *instructions* `(:add :mul))

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
  (unless (find instruction *instructions*)
    (error "unsupported instruction: ~a" instruction))

  (unless (assure-tensors variables)
    (error "all inputs must have same backends and be waffe tensor"))

  (let* ((backend (slot-value (first variables) 'backend))
	 (args (map 'list (lambda (x) (slot-value x 'data)) variables))
	 (result (case backend
		   (:cpu (cl-waffe.backends.cpu:kernel instruction args))
		   (:opencl (cl-waffe.backends.opencl:kernel instruction args)))))
    (const result backend)))

(defun backends-available ())
