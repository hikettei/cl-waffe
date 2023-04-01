
(in-package :cl-waffe)

(defnode SumTensor (axis)
  :optimize t
  :parameters ((axis axis) (repeats T))
  :forward ((x)
	    (setf (self repeats) (assure-tensor (!shape x (self axis))))
	    (with-searching-calc-node :sum x (self axis)))
  :backward ((dy)
	     (list (!div (!repeats dy (self axis) (self repeats))
			 (self repeats)))))

(defnode MeanTensor (axis)
  :optimize t
  :parameters ((axis axis) (repeats T))
  :forward ((x)
	    (setf (self repeats) (assure-tensor (!shape x (self axis))))
	    (with-searching-calc-node :mean x (self axis)))
  :backward ((dy) (list (!repeats dy (self axis) (self repeats)))))

(defnode SumUpTensor ()
  :parameters ((total-len) (shape))
  :forward ((x) ; only for 2d
		(setf (self total-len) (/ (!size x)))
		(setf (self shape) (!shape x))
		(sumup-tensor x))
  :backward ((dy)
	     (list (sysconst (scal! (self total-len)
				    (make-mat (self shape)
					      :initial-element (data dy)))))))

(defun !sum-2d (x &optional (axis nil) (keepdims nil))
  (if (null axis)
      (call (SumUpTensor) (assure-tensor x))
      (let ((nrepeat (!shape x axis))
	    (result (call (SumTensor (assure-tensor axis)) (assure-tensor x))))
	(if keepdims
	    (!repeats result axis nrepeat)
	    result))))

(defun !sum (x &optional (axis nil) (keepdims nil))
  "Sum up x where x is a cl-waffe tensor.

For nd tensors...
@begin(deflist)
@def(1D)
@term(unsqueeze x with 1, and call !sum again.)
@def(2D and more.)
@term(Sum up all elements of X)
@end(deflist)

@begin(section)
@title(arguments)

@begin(deflist)
@def(axis)
@term(a dimension to reduce)
@def(keepdims)
@term(When t, the returning tensor is repeated with @cl:param(axis))
@end(deflist)
@end(section)

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!randn `(10)))
(!sum a)
;=>#Const(4.74653)

(setq a (!randn `(10 10)))
(!sum a)
;=>#Const(1.5428619)

(!sum a 0)
;=>#Const(((-2.07... 0.463... ~ 1.778... 1.695...)) :mgl t :shape (1 10))

(!sum a 1)
;#Const(((0.967...)        
;                 ...
;        (2.774...)) :mgl t :shape (10 1))

(!sum a 0 t)
;#Const(((-2.07... 0.463... ~ 1.778... 1.695...)        
;                 ...
;        (-2.07... 0.463... ~ 1.778... 1.695...)) :mgl t :shape (10 10))
@end[lang=lisp](code)
@end(section)
"

  (declare (type (or null fixnum) axis)
	   (type boolean keepdims)
	   (type waffetensor x))
  (case (!dims x)
    (0 (error "!sum: the tensor given is a number"))
    (1 (!sum-2d x axis keepdims))
    (2 (!sum-2d x axis keepdims))
    (T
     (if (null axis)
	 (call (SumUpTensor) x)
	 (let* ((dims (!shape x axis))
		; Note: keepdims is ignored. And May need exclusive kernel for it because its too slow when forward and backward.

		(sum-dims #'(lambda (n) (loop for i upfrom 0 below (!dims x)
	 				      collect (if (= i axis)
							  n
							  t))))
		(result (!zeros (!shape (apply #'!aref x (funcall sum-dims 0))))))
	   (dotimes (i dims)
	     (setq result (!add result (apply #'!aref x (funcall sum-dims i)))))
	   result)))))


(defun !mean (x &optional (axis nil) (keepdims nil))
  "The usage is the same as !sum.

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!ones '(10 10)))
;#Const(((1.0 1.0 ~ 1.0 1.0)        
;                 ...
;        (1.0 1.0 ~ 1.0 1.0)) :mgl t :shape (10 10))
(!mean a)
;=>Const(1.0)
@end[lang=lisp](code)
@end(section)"
  (if (null axis)
      (!div (!sum x axis keepdims) (apply #'* (!shape x)))
      (!div (!sum x axis keepdims) (!shape x axis))))

