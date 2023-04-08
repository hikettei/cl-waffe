
(in-package :cl-waffe)

#|
  Here's Indexing APIs.
|#


(defun !view (tensor &rest subscript)
  "This is a basic function of Indexing.
This function gives tensor a subscript information."
  
  )



(defun get-stride (shape dim)
  (let ((subscripts (fill-with-d shape dim)))
    (apply #'+ (maplist #'(lambda (x y)
			    (the fixnum
				 (* (the fixnum (car x))
				    (the fixnum (apply #'* (cdr y))))))
			subscripts
			shape))))

(defmacro with-view (tensor var &body body)
  "var - the ref of tensor."
  `(let (())

     ))

