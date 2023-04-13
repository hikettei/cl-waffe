
(in-package :cl-waffe.impls.mps)

(define-node-extension cl-waffe::MatMulTensor
  :backend :mps
  :forward-declaim (declaim (ftype (function (cl-waffe::MatmulTensor waffetensor waffetensor) waffetensor) :forward))
  :forward ((x y)
	    (declare (optimize (speed 3) (safety 0)))
	    
	    (save-for-backward cl-waffe::xi x)
	    (save-for-backward cl-waffe::yi y)
	    
	    (setf (self cl-waffe::transpose-x?) (lazy-transpose-p x))
	    (setf (self cl-waffe::transpose-y?) (lazy-transpose-p y))
	    
	    (unless (= (!dims x) (!dims y) 2)
	      (error ""))

	    (let ((out (!zeros `(,(!shape x 0)
				 ,(!shape y 1)))))
	      (matmul-mps
	       x
	       y
	       out
	       :transpose-a (self cl-waffe::transpose-x?)
	       :transpose-b (self cl-waffe::transpose-y?))
	      out))
  :backward ((dy)
	     (list (!matmul dy (if (self cl-waffe::transpose-y?)
				   (progn
				     (value (self cl-waffe::yi) :ignore-transpose t)
				     (self cl-waffe::yi))
				   (!transpose (self cl-waffe::yi))))
		   (!matmul (if (self transpose-x?)
				(const (value (self cl-waffe::xi) :ignore-transpose t))
				(!transpose (self cl-waffe::xi)))
			    dy))))
