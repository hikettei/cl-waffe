
(in-package :cl-waffe.backends.cpu)

(defun repeat (array n &key axis)
  (if (numcl:arrayp array)
      (if axis
          (numcl:concatenate (make-list n :initial-element array) :axis axis)
          (numcl:flatten
           (numcl:concatenate (make-list n :initial-element (numcl:reshape array `(,@(numcl:shape array) -1))) :axis -1)))
      (progn
        ;(assert (null axis))
        (numcl:full n array))))

(defun kernel (ope args out)
  (declare (ignore out))
  ; (print "CPU Calling...") (print args) (print ope)
  (case ope
      (:add (+ (car args) (second args)))
      (:sub (- (car args) (second args)))
      (:mul (* (car args) (second args)))
      (:div (/ (car args) (second args)))
      (:log (log (car args)))
      (:exp (exp (car args)))
      (:pow (expt (car args) (second args)))
      (:sum (numcl:sum (car args) :axes (second args)))
      (:mean (numcl:mean (car args) :axes (second args)))
      (:tanh (tanh (car args)))
      (:repeat (repeat (car args) (third args) :axis (second args)))
      (:transpose (numcl:transpose (car args) (second args)))
      (T (error "~a is nt yet implemented" ope))))

(defun infomation ())
