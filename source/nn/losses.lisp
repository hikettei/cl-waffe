
(in-package :cl-waffe.nn)

(defun mse (p y) ; PowBackward...
  (mean (pow (sub p y) 2) 0))

(defun cross-entropy (x y &optional (delta 1e-7)) ; not supporting mini-batch
  (mul -1 (sum (sum (mul y (loge (add x delta))) 1) 0)))

