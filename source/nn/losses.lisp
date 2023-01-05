
(in-package :cl-waffe.nn)

(defun mse (p y) ; powbackward?
  (!mean (!pow (!sub p y) 2) 1))

(defun cross-entropy (x y &optional (delta 1e-7))
  (!mul -1 (!sum (!mul y (!log (!add x delta))) 1)))

