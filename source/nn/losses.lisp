
(in-package :cl-waffe.nn)

(defun mse (p y)
  (!mean (!pow (!sub p y) 2) 1))

(defun cross-entropy (x y &optional (delta 1e-7))
  ; x...
  ; y ... (batch-size n-classes)
  (!div (!mul -1 (!sum (!mul y (!log (!add x delta))))) (!shape y 0)))


