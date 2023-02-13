
(in-package :cl-waffe-test)

; utils for testing

(defun ~= (x y)
  (< (abs (- x y)) 0.00001))

(defun ~=1 (x y)
  (< (abs (- x y)) 1e-2))


(defmacro with-operate (x &key mgl waffe))
