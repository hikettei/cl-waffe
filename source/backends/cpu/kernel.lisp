
(in-package :cl-waffe.backends.cpu)

(deftype WaffeSupportedDataType ()
  `(or fixnum float null cons ratio))

(defun repeat (array n &key axis)
  ; asserted array is not tensor and may be axis is always zero
  (let ((dims (case axis
		(0 `(,n))
		(1 `(1, n))
		(T (error "kernel error")))))
    (mgl-mat:make-mat dims :initial-element array)))

(declaim (ftype (function (cons) cons) assure-args))
(defun assure-args (args)
  (declare (optimize (speed 3) (safety 0) (debug 0))
	   (type cons args))
  (map 'list (lambda (x)
	       (if (typep x 'function)
		   (funcall x nil t)
		   x))
       args))

(declaim (ftype (function (keyword cons) waffesupporteddatatype) kernel))
(defun kernel (ope args)
  (declare (optimize (speed 3) (safety 0) (debug 0))
	   (type keyword ope)
	   (type cons args))  
  (let* ((args (assure-args args)))
  (case ope
      (:add (+ (car args) (second args)))
      (:sub (- (car args) (second args)))
      (:mul (* (car args) (second args)))
      (:div (/ (car args) (second args)))
      (:log (log (car args)))
      (:sqrt (sqrt (car args)))
      (:exp (exp (car args)))
      (:pow (expt (car args) (second args)))
      (:sum (numcl:sum (car args) :axes (second args)))
      (:mean (numcl:mean (car args) :axes (second args)))
      (:tanh (tanh (car args)))
      (:repeat (repeat (car args) (third args) :axis (second args)))
      ;(:transpose (numcl:transpose (car args) (second args)))
      (T (error "~a is nt yet implemented" ope)))))

(defun infomation ())
