
(in-package :cl-waffe.nn)

(defun mse (p y)
  (mean (pow (sub p y) 2)))

