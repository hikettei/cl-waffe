
(in-package :cl-waffe)

(defun plusns (tensor)
  (let* ((dims (shape tensor))
	 (res (data tensor))
         (len (if (listp dims) (reduce #'* dims) dims)))
    (loop for n from 0 to (1- len)
          do (setf (row-major-aref res n) (if (> (row-major-aref res n) (coerce 0 'double-float))
						 (row-major-aref res n)
						 (coerce 0 'double-float))))
    res))

(defmodel ReLUTensor nil
  :parameters ((path-through T))
  :forward ((x)
	    (setf (self path-through) (assure-tensor (numcl:asarray (plusns x))))
	    (callop :mul (self path-through) x))
  :backward ((dy) (list (mul (self path-through) dy))))

(defun relu (x)
  (call (ReLUTensor) (assure-tensor x)))

(defmodel SigmoidTensor nil
  :parameters ((xi T))
  :forward ((x)
	    (setf (self xi) x)
	    (div 1 (add 1.0 (t-exp (mul -1 x)))))
  :backward ((dy) (list (mul (sigmoid (sigmoid (self xi))) (mul dy (sub (ones-like (self xi) (sigmoid (sigmoid (self xi))))))))))

(defun sigmoid (x)
  (call (SigmoidTensor) (assure-tensor x)))

(defmodel TanhTensor nil
  :parameters nil
  :forward ((x)
	    (callop :tanh x))
  :backward ((dy)
	     (sub 1 (pow (callop :tanh dy) 2))))

(defun wf-tanh (x)
  (call (TanhTensor) (assure-tensor x)))
