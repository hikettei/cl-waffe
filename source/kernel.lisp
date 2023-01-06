
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
			       :matmul
			       :exp
			       :tanh
			       :reshape
			       :transpose
			       :repeat))

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
  (unless (find instruction *instructions*) ;doesnt works?
    (error "unsupported instruction: ~a" instruction))

  (unless (assure-tensors variables)
    (error "all inputs must have same backends and be waffe tensor"))
  
  (let* ((backward? (find t (map 'list (lambda (x) (waffetensor-backward-mode x)) variables)))
	 (carx (car variables))
	 (tmp nil)
	 (out (unless backward?
		(if (find t (map 'list (lambda (x) (typep x 'waffetensor)) variables))
		    (let ((r (find t variables :test (lambda (_ x) (declare (ignore _))
							       (if (typep (data x) 'mgl-mat:mat)
								   (waffetensor-out x))))))
		      (if r
			  (setq tmp (waffetensor-out r))
			  nil))
		    nil)
		nil))
	 (out tmp) ; disable with out=nil
	 (backend (waffetensor-backend (first variables)))
	 (args (map 'list (lambda (x) (waffetensor-data x)) variables))
	 (all-not-array (every (lambda (x) (typep x 'waffesupporteddatatype)) args))
	 (result (case backend
		   (:cpu    (cl-waffe.backends.cpu:kernel instruction args out))
		   (:opencl (cl-waffe.backends.opencl:kernel instruction args out))
		   (:mgl    (if all-not-array ; Use CPU When like Const(1) + Const(1)
			        (cl-waffe.backends.cpu:kernel instruction args out)
				(cl-waffe.backends.mgl:kernel instruction args out)))))
	 (result (if (numcl:numcl-array-p result) ; slow?
		     (mgl-mat:array-to-mat result)
		     result)))

    ;(if backward?
	;(dolist (v variables)
	 ; (print result) ; created new buffer 4
	  ;(print (grad-tmp-value (waffetensor-grad-tmp v)))))
        
    (if (typep result 'mgl-mat:mat)
	(unless backward?
	  (unless out
	    (dolist (var variables)
	      (setf (waffetensor-out var) result)))))
    (const result :backend backend)))
	       

(defun backends-available ())

(defun check-supported-instruction (backend))

