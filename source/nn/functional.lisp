
(in-package :cl-waffe.nn)

(defun linear (x weight bias)
  ; Applies a linear transformation to the coming datum. y = xA + b
  (if bias
      (!add (!matmul x weight) bias)
      (!matmul x weight)))

