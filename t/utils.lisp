
(in-package :cl-waffe-test)

; utils for testing

(defun ~= (x y)
  (< (abs (- x y)) 0.00001))

(defun ~=1 (x y)
  (< (abs (- x y)) 1e-2))


;(defmacro with-operate (x &key mgl waffe))

(defmacro maxlist (list)
  `(let ((max-item (apply #'max ,list)))
     (if (<= max-item 1.0) 1.0 max-item)))

(defun max-position-column (arr)
  (declare (optimize (speed 3) (space 0) (safety 0) (debug 0))
           (type (array single-float) arr))
  (let ((max-arr (make-array (array-dimension arr 0)
                             :element-type 'single-float
                             :initial-element most-negative-single-float))
        (pos-arr (make-array (array-dimension arr 0)
                             :element-type 'fixnum
                             :initial-element 0)))
    (loop for i fixnum from 0 below (array-dimension arr 0) do
      (loop for j fixnum from 0 below (array-dimension arr 1) do
        (when (> (aref arr i j) (aref max-arr i))
          (setf (aref max-arr i) (aref arr i j)
                (aref pos-arr i) j))))
    pos-arr))
