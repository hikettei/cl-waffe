
(in-package :cl-waffe)

(defstruct (WaffeTensor (:constructor
		       tensor
			    (value &optional (backend :cpu)
			     &aux (data value) (backend backend) (grad `(0 nil))))
		   (:constructor
		       const
		       (value &optional (backend :cpu)
			&aux (data value) (backend backend) (grad nil))))
  data grad-tmp backward backend grad variables state)

(defun data (tensor)
  (slot-value tensor 'data))

(defun is-tensor (grad)
  (typep grad 'WaffeTensor))

(defun repeat (tensor n)
  (map 'list
       (lambda (x)
	 (declare (ignore x))
	 (const n))
       (slot-value tensor 'variables)))

(defmacro nth-var (tensor n)
  `(nth ,n (slot-value ,tensor 'variables)))

(defmacro nth-tensor (tensor n s)
  ; the nth variavle of tensor
  `(slot-value (nth-var ,tensor ,n) ,s))
  
(defun backward (tensor)
  (if (slot-value tensor 'backward)
      (let ((state (slot-value tensor 'state))
	    (grad  (if (slot-value tensor 'grad-tmp)
		       (slot-value tensor 'grad-tmp)
		       (repeat tensor 1))))
	(let ((grads (apply (slot-value tensor 'backward) state grad)))
	  (dotimes (i (length (slot-value tensor 'variables)))
	    (if (nth-tensor tensor i 'grad-tmp)
		(setf (nth-tensor tensor i 'grad-tmp)
		      (repeat (nth-var tensor i) (data (add (nth-tensor tensor i 'grad-tmp) (nth i grads)))))
		(setf (nth-tensor tensor i 'grad-tmp) (repeat (nth-var tensor i) (data (nth i grads)))))
	    (if (nth-tensor tensor i 'grad)
		(if (typep (nth-tensor tensor i 'grad) 'cons)
		    (setf (nth-tensor tensor i 'grad) (data (nth i grads)))
		    (setf (nth-tensor tensor i 'grad) (data (add (nth-tensor tensor i 'grad)
							    (data (nth i grads)))))))
	    (backward (nth-var tensor i)))))
      (setf (slot-value tensor 'grad-tmp) (if (slot-value tensor 'grad)
					  (repeat tensor 0)
					  (repeat tensor 0)))))
