
(in-package :cl-waffe.nn)

(defparameter *weight-initializer* :xavier)

(defun init-weight (dim)
  "Initialize weights for model in response to *weight-initializer*.
   The returned tensor is trainable."
  (case *weight-initializer*
    (:xavier
     (parameter (!randn dim)))
    (T
     (error "cl-waffe.nn:init-weight: unknown weight initializer ~a" *weight-initializer*))))
