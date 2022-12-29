
(in-package :cl-waffe.nn)

(defun linear (x weight bias)
  ; Applies a linear transformation to the coming datum. y = xA^T + b
  (if bias
      (add (transpose (matmul x weight)) bias)
      (transpose (matmul x weight))))

