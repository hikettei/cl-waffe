
(in-package :cl-waffe)

(defstruct (WaffeTensor (:constructor
		       tensor
			(value &optional (backend :cpu) &aux (data value) (backend backend) (is-param t)))
		   (:constructor
		       const
			(value &optional (backend :cpu) &aux (data value) (backend backend) (is-param nil))))
  data grad backward backend is-param variables)

(defun data (tensor)
  (slot-value tensor 'data))

(defun is-state (grad)
  (typep grad 'WaffeTensor))

(defun backward (tensor)
  (dolist (i (slot-value tensor 'variables))
    (if (slot-value i 'is-param)
	(if (slot-value i 'backward)
	    (backward i)
	    (setf (slot-value i 'grad) (const 1)))
	(setf (slot-value i 'grad) (const 0))))

  (if (slot-value tensor 'variables)
      (let ((args (map 'list (lambda (x) (slot-value x 'grad))
		       (slot-value tensor 'variables))))
	(let ((grads (apply (slot-value tensor 'backward) (slot-value tensor 'grad) args)))
	  (dotimes (i (length (slot-value tensor 'variables))) ;error処理はさむ
	    (setf (slot-value
		   (nth i (slot-value tensor 'variables)) 'grad)
		   (nth i grads)))))
      nil))
