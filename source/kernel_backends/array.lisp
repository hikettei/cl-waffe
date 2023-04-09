
(in-package :cl-waffe.kernel)

(defun mgl-mat::ctype-size (ctype)
  (case ctype
    (:short 2)
    (:float 4)
    (:double 8)))

(defun mgl-mat::ctype->lisp (dtype)
  (case dtype
    (:short 'short-float)
    (:float 'single-float)
    (:double 'double-float)))

(defun mgl-mat::coerce-to-ctype (element &key (ctype *dtype*))
  (coerce element (dtype->lisp ctype)))


