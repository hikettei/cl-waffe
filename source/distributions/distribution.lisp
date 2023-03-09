
(in-package :cl-waffe)

(defun !bernoulli (dims rate)
  "Init a tensor of dims with bernoulli

rate is single-float, and [0 1]

See also: @cl:param(!binomial), alias for it.

Example:
@begin[lang=lisp](code)
(!binomial '(10 10) 0.5)
;#Const(((1.0 0.0 ~ 1.0 1.0)        
;                 ...
;        (0.0 1.0 ~ 1.0 0.0)) :mgl t :shape (10 10))
@end[lang=lisp](code)"
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
