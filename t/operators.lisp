
(in-package :cl-waffe-test)

; 基本関数のforward backwardを検証

(in-suite :test)

(setq a (randn 10 10))
(setq b (randn 10 10))

(test cl-waffe-test
      (is (= 6 (mul 3 (add 1 1))))
      (is (add a b))
      (is (sub a b)))

