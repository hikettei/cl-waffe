
(in-package :cl-waffe.impls.mps)

(define-node-extension cl-waffe::MatMulTensor
  :backend :mps
  :forward-declaim (declaim (ftype (function (cl-waffe::MatmulTensor waffetensor waffetensor) waffetensor) :forward))
  :forward ((x y)
	    (declare (optimize (speed 3) (safety 0)))
	    
	    (save-for-backward xi x)
	    (save-for-backward yi y)
	    
	    (setf (self transpose-x?) (lazy-transpose-p x))
	    (setf (self transpose-y?) (lazy-transpose-p y))
	    
	    (unless (= (!dims x) (!dims y) 2)
	      (error ""))

	    (let ((out (!zeros `(,(!shape x 0)
				 ,(!shape y 1)))))
	      (matmul-mps
	       x
	       y
	       out
	       :transpose-a (self transpose-x?)
	       :transpose-b (self transpose-y?))
	      out))
  :backward ((dy)
	     (list (!matmul dy (if (self transpose-y?)
				   (progn
				     (value (self yi) :ignore-transpose t)
				     (self yi))
				   (!transpose (self yi))))
		   (!matmul (if (self transpose-x?)
				(const (value (self xi) :ignore-transpose t))
				(!transpose (self xi)))
			    dy))))
