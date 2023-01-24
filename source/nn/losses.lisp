
(in-package :cl-waffe.nn)

(defmacro assure-fixnum (val)
  `(multiple-value-bind (f _) (round ,val)
     (declare (ignore _))
     f))

(defun mat-labels (base-vec label epsilon)
  (let ((v (!fill (!shape base-vec 2) epsilon)))
    (setf (!aref v (the fixnum
			(assure-fixnum
			 (mgl-mat:mref (data label) 0 0))))
	  (const (make-array '(1)
			     :initial-element (- 1 epsilon))))
    (!unsqueeze v)))

(defun to-onehot (ps vec epsilon)
  (let ((result (!fill (!shape ps) epsilon)))
    (loop for batch upfrom 0 below (!shape ps 0)
	  do (loop for i upfrom 0 below (!shape ps 1)
		   do (setf (!aref result batch i)
			    (mat-labels
			     ps
			     (!aref vec batch i)
			     epsilon))))
    result))

(defun mse (p y)
  (!mean (!pow (!sub p y) 2) 1))

(defun cross-entropy (x y &optional (delta 1e-7) (epsilon 0.0))
  ; epsilon ... an parameter for label smoothing
  ; Regards Correct=1-epsilon, Incorrect=epsilon
  ; x...
  ; y ... (batch-size n-classes)

  (if (> (!dims x) (!dims y))
      ; When Given y is not a onehot.
      (!div (!mul -1 (!sum (!mul (to-onehot x y epsilon) (!log (!add x delta))))) (!shape y 0))
      ; When Given y is a onehot.
      (!div (!mul -1 (!sum (!mul y (!log (!add x delta))))) (!shape y 0))))


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

