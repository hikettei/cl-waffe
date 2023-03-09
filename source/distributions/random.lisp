
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
  "Initialize an tensor of dims (cons)

!random can be called with a varying number of type of arguments:

@begin(section)
@title(When limit=fixnum)
init within the range of @c(0<=x<limit)

@begin[lang=lisp](code)
;#Const(((1.0 2.0 ~ 2.0 1.0)        
;                 ...
;        (2.0 2.0 ~ 2.0 2.0)) :mgl t :shape (10 10))
@end[lang=lisp](code)
@end(section)

@begin(section)
@title(When limit=single-float)
init within the range of @c(0<=x<limit)
@begin[lang=lisp](code)
(!random '(10 10) 3.0)
;#Const(((0.152... 2.203... ~ 2.360... 2.216...)        
;                 ...
;        (1.003... 2.257... ~ 2.305... 2.025...)) :mgl t :shape (10 10))
@end[lang=lisp](code)
@end(section)

@begin(section)
@title(When limit=(cons single-float1 single-float2))
init with single-float1<=x<single-float2, where each element is single-float.
@begin[lang=lisp](code)
(!random '(10 10) '(1.0 3.0))
;#Const(((1.982... 1.526... ~ 1.388... 1.312...)        
;                 ...
;        (1.829... 2.676... ~ 1.226... 2.980...)) :mgl t :shape (10 10))
@end[lang=lisp](code)
@end(section)

Return: WaffeTensor
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
  "Initializes the tensor of dims. Each element is initialized with @cl:param(f) where f is a lambda exp and called with index.

Warning: Using mref and slow algorithm, @b(it is so slow).

Example:
@begin[lang=lisp](code)
(!random-with '(10 10) #'(lambda (n) n))
;#Const(((0.0 1.0 ~ 8.0 9.0)        
;                 ...
;        (90.0 91.0 ~ 98.0 99.0)) :mgl t :shape (10 10))
@end[lang=lisp](code)

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


(defun !normal (dims &optional (mean 2.0) (var 1.0))
  "Init with normal distribution.

Warning: Using mref and slow algorithm, @b(its sooo slow.)

It is recommended to use !randn and transform it instead."
  (let* ((res (!zeros dims))
         (len (if (listp dims) (reduce #'* dims) dims)))
    (loop for n from 0 to (1- len)
          do (setf (!row-major-aref res n) (gaussiandb-random var mean)))
    res))

(defun !randn (dims)
  "Initializes tensor with normal distribution in a faster way where mean=0.0, var=1.0.

Example:

@begin[lang=lisp](code)
(!randn `(10 10))
;#Const(((0.677... 0.054... ~ 0.257... 0.261...)        
;                 ...
;        (0.063... 0.607... ~ 0.460... 0.730...)) :mgl t :shape (10 10))
@end[lang=lisp](code)"
  (!normal dims 0 1))
