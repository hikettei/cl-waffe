
(in-package :cl-waffe)

(defun !bernoulli (dims rate)
  "Initializes the tensor of dims with sampling bernoulli distribution, where p=rate. p=[0, 1]"
  (declare (optimize (speed 3))
	   (type cons dims)
	   (type (single-float 0e0) rate))
  (unless (<= rate 1.0)
    (error "!bernoulli: rate must be in the range of [0 1]"))
  (!modify (!zeros dims) :bernoulli (const rate)))

(declaim (inline !binomial))
(defun !binomial (dims rate)
  "Alias for !bernoulli"
  (!bernoulli dims rate))
