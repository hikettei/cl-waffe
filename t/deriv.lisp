
(in-package :cl-waffe-test)

(in-suite :test)

(setq a (tensor 5))
(setq z (add 2 (mul a 3)))
(print z)
(backward z)
(print z)
(print a)

; 3a+2 y'=3
