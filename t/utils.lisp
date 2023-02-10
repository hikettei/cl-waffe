
(in-package :cl-waffe-test)

; utils for testing

(defun ~= (x y)
  (< (abs (- x y)) 0.00001))


(defmacro with-operate (x &key mgl waffe))
