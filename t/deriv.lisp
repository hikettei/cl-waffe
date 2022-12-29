
(in-package :cl-waffe-test)

(in-suite :test)

(setq a (tensor 10))
(setq b (tensor 9))
(setq z (add (mul 3 a) (add (mul 3 a) (mul 2 b))))
;(print z)
(backward z)

;(print z)
(print a)
(print b)

; 3a+2 y'=3
