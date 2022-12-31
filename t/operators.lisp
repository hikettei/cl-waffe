
(in-package :cl-waffe-test)

; 基本関数のforward backwardを検証

(in-suite :test)

(setq a (parameter (randn 7 10)))
(setq b (randn 7 10))

(setq c (parameter (randn 8 3)))
(print c)

(test cl-waffe-test
      (is (= 6 (mul 3 (add 1 1))))
      (is (add a b))
      (is (sub a b)))
