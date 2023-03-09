
(in-package :cl-waffe)

(defun range-tail (start end acc step)
  (if (> start end)
      acc
      (range-tail start (- end step) (cons end acc) step)))

(defun range (start end &optional (step 1) (fill nil))
  (range-tail start end '() step))
#|
(defnode FilteringTensorBackward (dim iter-num args)
  :parameters ((dim dim :type fixnum)
	       (iter-num iter-num :type fixnum)
	       (args args :type cons))
  :forward ((&rest result)
	    result)
  :backward ((dy)
	     (loop for i fixnum upfrom 0 below (self iter-num)
		   collect (apply #'!aref `(,(self args) ,i)))))

(defun !filter-tensor (tensor dim batch-size function)
  "This is a intrinsical function of doing iteration for waffe-tensor.
This is the very fastst but not useful. So use macros in order to make it more useful."
  (let ((args (loop for i fixnum upfrom 0 below (max (1- dim) 0)
		    collect t)))
    (print args)
    (apply #'call `(,(FilteringTensorBackward
		      dim
		      ,(/ (!shape tensor dim) batch-size)))
	   (loop for i fixnum
		 upfrom 0
		   below (/ (!shape tensor dim) batch-size)
		 collect
		 (funcall function
			  i
			  (apply #'!aref tensor `(,@args ,i))))))
|#
(defun !dotensor () "")
(defun !displace () "")

