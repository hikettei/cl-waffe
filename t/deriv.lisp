
(in-package :cl-waffe-test)

(in-suite :test)

(setq a (parameter (const 10)))
(setq b (tensor 9))
(setq z (add (mul 3 a) (add (mul 3 a) (mul 2 b))))

(backward z)

(test cl-waffe-test
      (is (= (grad a) 6))
      (is (= (grad b) 2)))

