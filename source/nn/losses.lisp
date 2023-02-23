
(in-package :cl-waffe.nn)

(defmacro assure-fixnum (val)
  `(multiple-value-bind (f _) (round (the single-float ,val))
     (declare (ignore _))
     f))

(defun mat-labels (base-vec label epsilon)
  (declare ;(optimize (speed 3))
	   (type single-float epsilon))
  (let ((v (!fill `(,(!shape base-vec 2)) epsilon)))
    (setf (!aref v (the integer
			(assure-fixnum
			 (mgl-mat:mref (data label) 0 0))))
	  (const (mgl-mat:make-mat '(1)
				   :initial-element (- 1 epsilon))))
    (!unsqueeze (!unsqueeze v))))

(defun to-onehot (ps vec epsilon)
  "ps ... an tensor of possibilites, vec = labels epsilon=for smooth labeling."
  (declare (optimize (speed 3)))
  (let ((result (!zeros (!shape ps))))
    (loop for batch fixnum upfrom 0 below (!shape ps 0)
	  do (loop for i fixnum upfrom 0 below (!shape ps 1)
		   do (let ((classes (mat-labels
				      ps
				      (!aref vec batch i)
				      epsilon)))
			(setf (!aref result batch i) classes))))
    result))

(defun mse (p y)
  "MSE Loss"
  (!mean (!pow (!sub p y) 2) 1))

(defun cross-entropy (x y &optional (delta 1e-7) (epsilon 0.0))
  "CrossEntropy Loss"
  (declare (optimize (speed 3)))
  ; epsilon ... an parameter for label smoothing
  ; Regards Correct=1-epsilon, Incorrect=epsilon
  ; x...
  ; y ... (batch-size n-classes)

  ; Todo: Implement it on mgl kernel

  (if (> (!dims x) (!dims y))
      ; When Given y is not a onehot.
      (!div (!mul -1
		  (!sum (!mul (to-onehot x y epsilon)
			      (!log (!add x delta)))))
	    (!div (!size x) (!shape x -1)))
      ; When Given y is a onehot.
      (!div (!mul -1
		  (!sum (!mul y (!log (!add x delta)))))
	    (!div (!size x) (!shape x -1)))))

(defnode SoftMaxCrossEntropy (&key (delta 1e-7) (avoid-overflow t))
  :parameters ((delta delta) (avoid-overflow avoid-overflow) (batch-size T) (out T) (target T))
  :forward ((x y)
	    "x: weights (batch len classes) y: (batch len classes)"
	    (setf (self batch-size) (!shape y 0))
	    (save-for-backward target y)
	    (let ((z (!softmax x :avoid-overflow (self avoid-overflow))))
	      (save-for-backward out z)
	      (cross-entropy z y (self delta))))
  :backward ((dy)
	     (let* ((z (!sub (self out) (self target)))
		    (dx (!mul dy (!div z (self batch-size)))))
	       (list dx dx))))

(defun softmax-cross-entropy (x y &key (avoid-overflow t) (delta 1e-7) (epsilon 0.0))
  "Softmax-Cross-Entropy Loss"
  (if (> (!dims x) (!dims y))
      (call
       (SoftMaxCrossEntropy :avoid-overflow avoid-overflow :delta delta)
       x
       (to-onehot x y epsilon))
      (call (SoftMaxCrossEntropy :avoid-overflow avoid-overflow :delta delta) x y)))

