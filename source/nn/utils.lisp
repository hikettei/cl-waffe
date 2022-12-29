
(in-package :cl-waffe.nn)

(defmodel LinearLayer (in-features out-features &optional (bias T))
  :parameters ((weight (parameter (* 0.01 (randn in-feeatures out-features))))
	      (bias (if bias
			(parameter (zeros out-features))
			nil)))
  :forward ((x) (cl-waffe.nn:linear x (self weight) (self bias))))


