
(in-package :cl-waffe.nn)

(defun mse (p y)
  (!mean (!pow (!sub p y) 2) 1))

(defun cross-entropy (x y &optional (delta 1e-7))
  ; x...
  ; y ... (batch-size n-classes)
  (!div (!mul -1 (!sum (!mul y (!log (!add x delta))))) (!shape y 0)))


(defnode SoftMaxCrossEntropy (&key (delta 1e-7) (avoid-overflow t))
  :parameters ((delta delta) (avoid-overflow avoid-overflow) (batch-size T) (out T) (target T))
  :forward ((x y)
	    (setf (self batch-size) (!shape y 0))
	    (save-for-backward target y)
	    (let ((z (!softmax x :avoid-overflow (self avoid-overflow))))
	      (setf (self out) z)
	      (cross-entropy z y (self delta))))
  :backward ((dy)
	     (let* ((z (!sub (self out) (self target)))
		    (dx (!mul dy (!div z (self batch-size)))))
	       (list dx dx))))

(defun softmax-cross-entropy (x y &key (avoid-overflow t) (delta 1e-7))
  ; Todo For Batched.
  (call (SoftMaxCrossEntropy :avoid-overflow avoid-overflow :delta delta) x y))

