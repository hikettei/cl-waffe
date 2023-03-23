
(in-package :cl-waffe)

(defparameter *dtypes* `(:float)) ; To Add: :half

(defun dtype-p (dtype)
  (if (find dtype *dtypes*)
      dtype
      (error "Invaild dtype ~a. Dtype must be chosen by following: ~a"
	     dtype
	     *dtypes*)))

(defmacro with-dtype (dtype &body body)
  `(let ((mgl-mat:*DEFAULT-MAT-CTYPE* ,(dtype-p dtype)))
     ,@body))
