
(in-package :cl-waffe.backends.mgl)


(mgl-mat:define-lisp-kernel (adam-stepm-lisp)
    ((m :mat :io)
     (mgrads :mat :input)
     (beta single-float)
     (size fixnum))
  (loop for i fixnum upfrom 0 below size
	do (setf (aref m i) (+ (aref m i)
			       (* (- (aref mgrads i)
				     (aref m i))
				  ( - 1 beta))))))

(mgl-mat:define-lisp-kernel (adam-stepv-lisp)
    ((v :mat :io)
     (vgrads :mat :input)
     (beta single-float)
     (size fixnum))
  (loop for i fixnum upfrom 0 below size
	do (setf (aref v i) (+ (aref v i)
			       (* (- 1 beta)
				  (- (expt (aref vgrads i) 2) (aref v i)))))))

(mgl-mat:define-lisp-kernel (adam-step-grads)
    ((params :mat :io)
     (m :mat :input)
     (v :mat :input)
     (eps single-float)
     (lr-t single-float)
     (size fixnum))
  (loop for i fixnum upfrom 0 below size
	do (setf (aref params i)
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
  (adam-stepm-lisp m paramgrads beta1 matsize)
  (adam-stepv-lisp v paramgrads beta2 matsize)
  (adam-step-grads param m v epsilon lr-t matsize))
