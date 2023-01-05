
(in-package :cl-waffe-test)

(in-suite :test)

(setq a (parameter (const 10)))
(setq b (tensor 9))

(setq c (!mul 4 a))
(setq d (!mul 5 b))

(setq z (!add (!mul 7 a) (!add (!mul 7 c) (!mul 11 d))))

(backward z)

; c ... 7
; d ... 11

(print (grad a)) ; 7 + 28 = 35
(print (grad b)) ; 7 * 11 = 77

(test cl-waffe-test
      (is (= (grad a) 6))
      (is (= (grad b) 2)))

