
(in-package :cl-waffe.impls.mps)

(define-node-extension cl-waffe::SinTensor
    :backend :mps
    :forward ((x)
  	      (save-for-backward xi x)
	      (!sin x))
    :backward ((dy)
	       (list (!mul dy (!cos (self xi))))))
