
(in-package :cl-waffe.nn)

(defun linear (x weight bias)
  "Applies a linear transformation to the coming datum. y = xA + b"
  ;Todo: rewrite it with geem! to make faster
  (if bias
      (!add (!matmul x weight) bias)
      (!matmul x weight)))
