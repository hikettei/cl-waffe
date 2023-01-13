
(in-package :cl-waffe.backends.cpu)

(deftype ResultType ()
    `(VALUES
      (OR FIXNUM FLOAT (COMPLEX SINGLE-FLOAT) (COMPLEX DOUBLE-FLOAT))
      &OPTIONAL))


(defun repeat (array n &key axis)
  ; asserted array is not tensor and may be axis is always zero
  (let ((dims (case axis
		(0 `(,n))
		(1 `(1, n))
		(T (error "kernel error")))))
    (mgl-mat:make-mat dims :initial-element array)))

(declaim (ftype (function (cons) cons) assure-args))
(defun assure-args (args)
  (map 'list (lambda (x)
	       (declare (type waffetensor x))
	       (typecase (data x)
		 (function (funcall (data x) nil t))
		 (T (data x))))
       args))

(declaim (ftype (function (keyword cons) ResultType) kernel))
(defun dispatch-kernel (ope args)
  (let* ((args (assure-args args)))
  (case ope
      (:add (+ (car args) (second args)))
      (:sub (- (car args) (second args)))
      (:mul (* (car args) (second args)))
      (:div (/ (car args) (second args)))
      (:inv (/ 1 (car args)))
      (:log (log (car args)))
      (:sqrt (sqrt (car args)))
      (:exp (exp (car args)))
      (:pow (expt (car args) (second args)))
      (:tanh (tanh (car args)))
      (:repeat (repeat (car args) (third args) :axis (second args)))
      ;(:transpose (numcl:transpose (car args) (second args)))
      (T (error "~a is nt yet implemented" ope)))))


(defun infomation ())
