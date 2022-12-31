
(in-package :cl-waffe.optimizers)

(defoptimizer SGD (params &key (lr 1e-3))
  :parameters ((params params) (lr lr))
  :update (()
	   (dolist (p (self params))
	     (setf (data p) (data (sub p (mul (self lr) (grad p))))))))


