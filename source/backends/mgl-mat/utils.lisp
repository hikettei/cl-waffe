
(in-package :cl-waffe.backends.mgl)

(define-lisp-kernel (write-to-nth-dim-with-range-lisp)
    ; INDEX=(!shape tensor 0) * (x1) + (!shape tensor 1) * (x2) + ...
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

(define-lisp-kernel (write-to-nth-dim-with-range-lisp1)
    ((out :mat :io)
     (copy-from-mat :mat)
     (size fixnum)
     (bias fixnum))
  (loop for oi of-type fixnum upfrom 0 below size
        do (setf (aref out (+ oi bias))
		 (aref copy-from-mat oi))))

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


(defun write-to-nth-dim-with-range1 (out
				     copy-from-mat
				     bias)
  (if t;(use-cuda-p out)
      (write-to-nth-dim-with-range-lisp1
       out
       copy-from-mat
       (mgl-mat:mat-size copy-from-mat)
       bias)))

(define-lisp-kernel (copy-elements-lisp)
    ((result :mat :output)
     (tensor :mat :input)
     
     (iter-for-n+1 fixnum) ;1d
     (iter-for-n fixnum)   ;2d
     
     (result-bias fixnum) ; 2d
     (tensor-bias fixnum) ; 2d
     
     (result-n-disp fixnum)
     (tensor-n-disp fixnum)

     (2d-dif-r fixnum)
     (2d-dif-t fixnum)
     
     (bias fixnum))
  (loop for m fixnum upfrom 0 below iter-for-n
	do (loop for i fixnum upfrom 0 below iter-for-n+1
		 do (setf (aref result (+ (+ result-bias
					     (the fixnum (* (+ result-n-disp m) 2d-dif-r)))
					  i))
			  (aref tensor (+
					(+ tensor-bias
					   (the fixnum (* (+ tensor-n-disp m) 2d-dif-t)))
					i))))))

(defun copy-elements (nth
		      result
		      tensor
		      
		      iter-for-n+1 ; 1d
		      iter-for-n   ; 2d
		      
		      result-bias ; 2d
		      tensor-bias ; 2d
		      
		      result-displacements
		      tensor-displacements
		      
		      result-n-displacements
		      tensor-n-displacements

		      bias)
  (declare (optimize (speed 3))
	   (type mat result tensor)
	   (type fixnum
		 nth
		 iter-for-n+1
		 iter-for-n
		 result-bias
		 tensor-bias
		 result-displacements
		 tensor-displacements
		 result-n-displacements
		 tensor-n-displacements

		 bias))
  (let ((r-dif (get-difference result nth))
	(t-dif (get-difference tensor nth)))
    (declare (type fixnum r-dif t-dif))
  (copy-elements-lisp
   result
   tensor
   (the fixnum (+ bias iter-for-n+1))
   iter-for-n
   (the fixnum (+ (the fixnum (* r-dif result-n-displacements)) result-bias)) ; add result-bias to move.
   (the fixnum (+ (the fixnum (* t-dif tensor-n-displacements)) tensor-bias))
   result-displacements
   tensor-displacements
   r-dif
   t-dif
   bias))
  nil)

