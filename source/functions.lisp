
(in-package :cl-waffe)

(defun plusns (tensor)
  (let* ((dims (shape tensor))
	 (res (data tensor))
         (len (if (listp dims) (reduce #'* dims) dims)))
    (loop for n from 0 to (1- len)
          do (setf (row-major-aref res n) (if (> (row-major-aref res n) (coerce 0 'double-float))
						 (row-major-aref res n)
						 (coerce 0 'double-float))))
    res))

(defmodel ReLUTensor nil
  :parameters ((path-through T))
  :forward ((x) (setf (self path-through) (assure-tensor (numcl:asarray (plusns x))))
		(callop :mul (self path-through) x))
  :backward ((dy) (list (mul (self path-through) dy))))

(defun relu (x)
  (call (ReLUTensor) (assure-tensor x)))

