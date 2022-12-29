
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

(defun kernel (ope args)
  (case ope
    (:add (numcl:+ (car args) (second args)))
    (:sub (numcl:- (car args) (second args)))
    (:mul (numcl:* (car args) (second args)))
    (:div (numcl:/ (car args) (second args)))
    (:sum (numcl:sum (car args) :axes (second args)))
    (:repeat (repeat (car args) (third args) :axis (second args)))
    (T (error "~a is nt yet implemented" ope))))

