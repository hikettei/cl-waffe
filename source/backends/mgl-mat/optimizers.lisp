
(in-package :cl-waffe.backends.mgl)

(mgl-mat:define-lisp-kernel (adam-step-grads)
    ((params :mat :output)
     (m :mat :output)
     (v :mat :output)
     (grads :mat :input)
     (eps single-float)
     (lr-t single-float)
     (beta1 single-float)
     (beta2 single-float)
     (size fixnum))
  (loop for i fixnum upfrom 0 below size
	do (setf (aref m i) (+ (aref m i)
			       (* (- (aref grads i)
				     (aref m i))
				  ( - 1 beta1))))
	   (setf (aref v i) (+ (aref v i)
			       (* (- 1 beta2)
				  (- (expt (aref grads i) 2) (aref v i)))))
	   (setf (aref params i)
		 (- (aref params i)
		    (/ (* lr-t (aref m i))
		       (+ eps (sqrt (the (single-float 0e0) (aref v i)))))))))

(defun adam-update (m
		    v
		    beta1
		    beta2
		    param
		    paramgrads
		    matsize
		    epsilon
		    lr-t)
  (adam-step-grads
   param
   m
   v
   paramgrads
   epsilon
   lr-t
   beta1
   beta2
   matsize))
