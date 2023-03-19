
(in-package :cl-waffe)

(defun !chisquare (dims df)
  "Initializes tensor with samples of chi-square distribution using the gamma distribution.
  Parameters:
  dims - The dimensions of the tensor.
  df   - The degrees of freedom of the chi-square distribution."
  (declare (optimize (speed 3))
           (type cons dims)
           (type single-float df))
  (!gamma dims (/ df 2.0) 2.0))
