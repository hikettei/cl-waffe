
(in-package :cl-waffe.nn)

(defun linear (x weight bias)
  ; Applies a linear transformation to the coming datum. y = xA + b
  (if bias
      (!add (!dot x weight) bias)
      (!dot x weight)))

