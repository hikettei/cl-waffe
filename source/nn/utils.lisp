
(in-package :cl-waffe.nn)

(defmodel LinearLayer (in-features out-features &optional (bias T))
  :parameters ((weight (parameter (mul 0.01 (randn in-features out-features))))
	      (bias (if bias
			(parameter (zeros out-features))
			nil)))
  :forward ((x) (cl-waffe.nn:linear x (cl-waffe:self weight) (cl-waffe:self bias))))


