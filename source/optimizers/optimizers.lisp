
(in-package :cl-waffe.optimizers)

(defoptimizer SGD (params &key (lr 1e-3))
  :parameters ((params params) (lr lr))
  :update (()
	   (dolist (p (self params))
	     (print (const (Grad p)))
	     (setf (data p) (data (!sub p (!mul (self lr) (grad p))))))))


