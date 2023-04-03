
(in-package :cl-waffe)

(defun double-random ()
  (let ((i (random 1.0)))
    (if (eq i 0.0)
	(setq i (double-random)))
    i))

(defun gaussiandb-random (var mean)
  (let* ((r (double-random))
	 (c (sqrt (* -2 (log r)))))
    (if (< (double-random) 0.5)
	(+    (* c
	      (sin (* 2.0 pi (double-random)))
	      var)
	      mean)
	(+    (* c
	      (cos (* 2.0 pi (double-random)))
	      var)))))

; Todo: Optimize
(defun !random (dims limit)
  "Initializes the new tensor of dims. Each element is consisted of a uniform-random within limit. limit must be following: fixnum, single-float, cons. and depending on this !random has a multiple behaviours.
"
  (let* ((res (!zeros dims))
         (upper-limit (if (listp limit) (second limit) limit))
         (lower-limit (if (listp limit) (first limit) 0))
         (len (if (listp dims) (reduce #'* dims) dims))
         (tmp-limit (- upper-limit lower-limit)))
    (loop for n from 0 to (1- len)
          do (setf (!row-major-aref res n)
                   (+ (random tmp-limit) lower-limit)))
    res))

(declaim (ftype (function ((or cons fixnum) function) waffetensor) !random-with))
(defun !random-with (dims f)
  "Initializes the tensor of dims. Each element is initialized with @cl:param(f), f is a funcallable function. and called with the index of the tensor.

See also: !init-with which is alias for !random-with.
"
  (declare (optimize (speed 3) (safety 0) (space 0))
	   (type function f))
  (let* ((res (make-array dims :initial-element 0))
         (len (the fixnum (if (listp dims) (reduce #'* dims) dims))))
    (loop for n fixnum from 0 to (1- len)
          do (setf (row-major-aref res n)
                   (funcall f n)))
    (const res)))

(declaim (inline !init-with))
(defun !init-with (dims f)
  "Alias for !random-with. This function is inlined."
  (!random-with dims f))


(defun !normal (dims &optional (mean 2.0) (stddev 1.0))
  "Initializes the new tensor with sampling the standard distribution."
  (declare (type cons dims))
  (let* ((res (!zeros dims)))
    (gaussian-random! (data res) :mean mean :stddev stddev)
    res))

(defun !randn (dims)
  "Initializes the new tensor of dims with sampling normal distribution where mean=0.0, stddev=1.0"
  (!normal dims 0.0 1.0))

(defun !uniform-random (dims &key (limit 1))
  "Initializes tensor with sampling uniform random.

The returned tensor is filled by random numbers 0<=x<limit"
  (let ((res (!zeros dims)))
    (uniform-random! (data res) :limit limit)
    res))
