
(in-package :cl-waffe.backends.mgl)

(define-lisp-kernel (write-to-nth-dim-with-range-lisp)
    ((out :mat :io)
     (copy-from-mat :mat)
     (target-axis fixnum)
     (start fixnum)
     (doeach-out fixnum)
     (doeach fixnum)
     (bias fixnum))
  (loop for oi of-type fixnum upfrom 0 below doeach-out
        do (setf (aref out (+ oi
			      (the fixnum
				   (* target-axis doeach-out))))
		 (aref copy-from-mat
		       (+ oi bias
			  (the fixnum
			       (* (the fixnum (+ start target-axis)) doeach)))))))

(defun fill-with-d (mat i n)
  (let ((index -1))
    (map 'list (lambda (x)
		 (declare (ignore x))
		 (incf index 1)
		 (cond
		   ((= i index)
		    n)
		   (T 0)))
	 (mat-dimensions mat))))

(defun get-difference (mat target-dim)
  "when defining lisp kernel, esp for 3d, 4d tensor. this is useful
   This function calculates that, the number that +1 in mat equivalent to in 1d mat"
  (- (apply #'mat-row-major-index mat (fill-with-d mat target-dim 1))
     (apply #'mat-row-major-index mat (fill-with-d mat target-dim 0))))

(defun write-to-nth-dim-with-range (out
				    copy-from-mat
				    target-dim
				    target-axis
				    start
				    bias)
  (if t;(use-cuda-p out)
      (write-to-nth-dim-with-range-lisp
       out
       copy-from-mat
       target-axis
       start
       (get-difference out target-dim)
       (get-difference copy-from-mat target-dim)
       bias))
  (get-difference copy-from-mat target-dim))
