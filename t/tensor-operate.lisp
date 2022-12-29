
(in-package :cl-waffe-test)

(in-suite :test)

(setq weight (parameter (randn 10 10)))
(setq x (randn 10 10))
(setq bias (randn 10 10))


(setq result (mean (add (mul weight x) bias) 0))

(print (data result))
(backward result)

;(print (grad weight))

