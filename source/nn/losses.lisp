
(in-package :cl-waffe.nn)

(defun mse (p y)
  (mean (pow (sub p y) 2)))

(defun cross-entropy (x y &optional (delta 1e-7))
  (let* ((coeff (div -1.0 (max (length (data x)) 1)))
	 (res (mul coeff (mul y (sum (log (add x delta)) 0))))) ; axis=0?
    res))


