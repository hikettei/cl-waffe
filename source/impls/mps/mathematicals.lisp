
(in-package :cl-waffe.impls.mps)

(define-node-extension cl-waffe::SinTensor
    :backend :mps
    :forward ((x)
  	      (save-for-backward xi x)
	      (let ((result (maybe-copy x)))
		(with-facets ((x* ((data x) 'foreign-array :direction :input))
			      (o* ((data result) 'foreign-array :direction :output)))
		  nil)
		result))
    :backward ((dy)
	       (list (!mul dy (!cos (self xi))))))
