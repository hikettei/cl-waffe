
(in-package :cl-waffe)

(defparameter *kernels* `(:cpu :opencl :mgl))
(defparameter *instructions* `(:add
			       :sub
			       :mul
			       :div
			       :log
			       :pow
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

(defparameter *keepdims-instructions* `(:add :sub :mul :div :log :pow :< :> :exp :tanh))

(defparameter *input-to-out-shape-table* (make-hash-table :test #'equal)) ; these values are also values of result...

(defmacro find-shape (instruction variables) ; reduce the num of keys
  `(gethash `(,(if (find instruction *keepdims-instructions* :test #'eq)
		  T
		  instruction)
	     ,@(map 'list (lambda (x)
		       (if (typep (data x) 'mgl-mat:mat)
			   (mgl-mat:mat-dimensions (data x))
			   (data x)))
		     variables))
	   *input-to-out-shape-table*))

(defnode 1dArrayToConstTensor nil
  :parameters ((xi T))
  :forward ((x)
	    (setf (self xi) x)
	    (const (mgl-mat:mref (data x) 0)))
  :backward ((dy)
	     (list (callop :mul dy (self xi)))))

(defun !1darray-to-const (x)
  (call (1dArrayToConstTensor) (assure-tensor x)))

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
  ;(declare (optimize (speed 3) (space 0) (safety 0) (debug 0)))
  (unless (find instruction *instructions*) ;doesnt works?
    (error "unsupported instruction: ~a" instruction))

  (unless (assure-tensors variables)
    (error "all inputs must have same backends and be waffe tensor"))
  
  (let* ((out (find-shape instruction variables))
	 (backend (waffetensor-backend (first variables)))
	 (args (map 'list (lambda (x) (waffetensor-data x)) variables))
	 (all-not-array (every (lambda (x) (typep x 'waffesupporteddatatype)) args))
	 (result (case backend
		   (:cpu    (cl-waffe.backends.cpu:kernel instruction args out))
		   ;(:opencl (cl-waffe.backends.opencl:kernel instruction args out))
		   (:mgl    (if all-not-array ; Use CPU When like Const(1) + Const(1)
			        (cl-waffe.backends.cpu:kernel instruction args out)
				(cl-waffe.backends.mgl:kernel instruction args out)))
		   (T (error "No such backends: ~a" backend))))
	 (res-tensor (const result :backend backend)))

    (if (typep result 'mgl-mat:mat)
	(unless out
	  (setf (find-shape instruction variables) result)))

        res-tensor))

(defun backends-available ())

(defun check-supported-instruction (backend))

