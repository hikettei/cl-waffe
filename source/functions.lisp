
(in-package :cl-waffe)

(defun plusns (tensor) ; gpu ver...?
  (let* ((dims (!shape tensor))
	 (res (data tensor))
         (len (if (listp dims) (reduce #'* dims) dims)))
    (loop for n from 0 to (1- len)
          do (setf (row-major-aref res n) (if (> (row-major-aref res n) (coerce 0 'double-float))
						 (row-major-aref res n)
						 (coerce 0 'double-float))))
    res))

(defnode ReLUTensor nil
  :parameters ((path-through T))
  :forward ((x)
	    (setf (self path-through) (assure-tensor (numcl:asarray (plusns x))))
	    (callop :mul (self path-through) x))
  :backward ((dy) (list (callop :mul (self path-through) dy))))

(defun !relu (x)
  (call (ReLUTensor) (assure-tensor x)))

(defnode SigmoidTensor nil
  :parameters ((xi T))
  :forward ((x)
	    (setf (self xi) x)
            (callop :div (!add 1 (!tanh (!div x 2))) (const 2)))
  :backward ((dy) (let ((p (!sigmoid (self xi))))
		    (list (callop :mul p (!mul dy (!sub 1 p)))))))

(defun !sigmoid (x)
  (call (SigmoidTensor) (assure-tensor x)))

(defnode TanhTensor nil
  :parameters nil
  :forward ((x)
	    (callop :tanh x))
  :backward ((dy)
	     (callop :sub (const 1) (!pow (callop :tanh dy) 2))))

(defun !tanh (x)
  (call (TanhTensor) (assure-tensor x)))


