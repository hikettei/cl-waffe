
(in-package :cl-waffe)

(defun !chisquare (dims df)
  "Initializes tensor with samples of chi-square distribution using the gamma distribution.."
  (declare (optimize (speed 3))
           (type cons dims)
           (type (single-float 0e0) df))
  
  (!gamma dims (/ df 2.0) 2.0))
