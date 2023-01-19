
(in-package :cl-waffe.backends.mgl)


(mgl-mat:define-lisp-kernel (adam-stepm-lisp)
    ((m :mat :io)
     (mgrads :mat :io)
     (beta single-float)
     (size fixnum))
  (loop for i upfrom 0 below size
	do (setf (aref m i) (+ (aref m i)
			       (* (- (aref mgrads i)
				     (aref m i))
				  ( - 1 beta))))))

(mgl-mat:define-lisp-kernel (adam-stepv-lisp)
    ((v :mat :io)
     (vgrads :mat :io)
     (beta single-float)
     (size fixnum))
  (loop for i upfrom 0 below size
	do (setf (aref v i) (+ (aref v i)
			       (* (- 1 beta)
				  (- (expt (aref vgrads i) 2) (aref v i)))))))

(mgl-mat:define-lisp-kernel (adam-step-grads)
    ((params :mat :io)
     (m :mat :io)
     (v :mat :io)
     (eps single-float)
     (lr-t single-float)
     (size fixnum))
  (loop for i upfrom 0 below size
	do (setf (the single-float (aref params i))
		 (- (the single-float (aref params i))
		    (/ (* lr-t (the single-float (aref m i)))
		       (+ eps (the single-float (sqrt (the single-float (aref v i))))))))))

(defun adam-update (m
		    v
		    beta1
		    beta2
		    param
		    paramgrads
		    matsize
		    epsilon
		    lr-t)
  (adam-stepm-lisp m paramgrads beta1 matsize)
  (adam-stepv-lisp v paramgrads beta2 matsize)
  (adam-step-grads param m v epsilon lr-t matsize))
