
(in-package :cl-waffe.nn)

(defparameter *weight-initializer* :xavier
  "Available Methods:
    :xavier (Default),
    :xavier-uniform,
    :xavier-LeCun,
    :xavier-uniform,

    :He
    :normal")

(defun select-initializer (activation)
  "Returns a property initialization method depending on activation. (Todo: More)"
  (case activation
    (:relu :he)
    (:sigmoid :xavier)
    (:tanh :xavier)
    (T nil)))

(defmacro init-activation-weights (activation in-features out-features)
  `(init-weights (select-initializer ,activation)
		 ,in-features
		 ,out-features))

(defun init-weights (method in-features out-features)
  "Initialize weights for model in response to *weight-initializer*.
   The returned tensor is trainable."
  (declare (type fixnum in-features out-features))
  (let ((dim `(,in-features ,out-features)))
    (case (or method
	      *weight-initializer*)
      (:xavier
       (parameter (!normal dim 0.0 (sqrt (/ (+ in-features out-features))))))
      (:xavier-LeCun
       (parameter (!normal dim 0.0 (sqrt (/ in-features)))))
      (:xavier-uniform
       (parameter (!!mul (sqrt (/ 6.0 (+ in-features out-features)))
			 (!random dim `(-1.0 1.0)))))
      (:he
       (parameter (!!mul (!randn dim) (sqrt (/ 2.0 in-features)))))
      (:normal
       (parameter (!normal dim 0 0.1)))
      (T
       ; Todo: if *weight-initializer* is a function, call it.
       (error "cl-waffe.nn:init-weights: unknown weight initializer ~a" (or method *weight-initializer*))))))
