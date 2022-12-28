
(in-package :cl-waffe.backends.cpu)

(defun kernel (ope args)
  (case ope
    (:add (+ (car args) (second args)))
    (:mul (* (car args) (second args)))))
