
(in-package :cl-waffe.optimizers)

(defoptimizer SGD (params &key (lr 1e-3))
  :parameters ((params params) (lr lr))
  :update (()
	   (dolist (p (self params))
	     (mgl-mat:copy! (data (!sub p (!mul (self lr) (grad p)))) (data p)))))


