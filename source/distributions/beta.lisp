
(in-package :cl-waffe)

; To fix: Zero-division error.
(defun !beta (dims alpha beta)
  "Initializes tensor with samples of beta distribution in a faster way.

Algorithm: https://dl.acm.org/doi/pdf/10.1145/359460.359482

x=[0,1]

a = min(alpha, beta)

b = max(alpha, beta)

PDF: fX(x)=x^a−1*(1−x)*b−1/B(a,b)

where B(a,b)=∫1,0{x^a−1(1−x)^b−1}dx

@begin[lang=lisp](code)
(time (!beta '(200) 5.0 1.0))
;Evaluation took:
;  0.000 seconds of real time
;  0.000063 seconds of total run time (0.000063 user, 0.000000 system)
;  100.00% CPU
;  143,846 processor cycles
;  0 bytes consed
  
;#Const((0.813... 0.832... ~ 0.865... 0.787...) :mgl t :shape (200))
@end[lang=lisp](code)"

  (declare (optimize (speed 3))
	   (type cons dims)
	   (type single-float alpha beta))
  (let* ((a (min alpha beta))
 	 (b (max alpha beta))
	 (result (!zeros dims))
	 (size (!size result)))
    (declare (type fixnum size))
    (with-facet (array ((data result) 'backing-array :direction :output))
      (declare (type (simple-array single-float) array))
      ; Todo For GPU.
      (loop for i fixnum upfrom 0 below size
	    do (setf (aref array i)
		     (if (> a 1)
			 (!beta-bb alpha a b)
			 (!beta-bc alpha a b)))))
    result))


(declaim (ftype (function
		 (single-float single-float single-float)
		 single-float)
		!beta-bb
		!beta-bc))
(defun !beta-bb (a0 a b)
  "Generates beta variances.

Algorithm: https://dl.acm.org/doi/pdf/10.1145/359460.359482

Note: !beta excepts that @c((min a b) > 1)"
  (declare (optimize (speed 3) (safety 0) (debug 0))
	   (type single-float a0)
	   (type (single-float 0e0) a b))

  (unless (> (min a b) 1.0)
    (error "cl-waffe:!beta failed because of (min a b) > 1."))

  (let* ((alpha (+ a b))
  	 (beta  (sqrt (the (single-float 0e0)
			   (/ (- alpha 2.0)
			      (- (* 2.0 a b) alpha)))))
	 (gamma (+ a (/ beta)))
	 (r0 0.0)
	 (w0 0.0)
	 (t0 0.0))
    (labels ((next (&aux
		      (u1 (random 1.0))
		      (u2 (random 1.0))
		      (v (* beta (- (log u1) (log (+ 1.0 (- u1)))))))
	       (declare (type single-float u1 u2 v))
	       
	       (setq w0 (* a (exp v)))
	       (setq r0 (- (* gamma v) 1.3862944))
	       
	       (let* ((z (* u1 u1 u2))
		      (s (+ a r0 (- w0))))
		 (declare (type single-float z s))
		 
		 (if (>= (+ s 2.609438) (* 5 z))
		     nil
		     (progn
		       (setq t0 (log z))
		       (if (>= s t0)
			   nil
			   t))))))
      (loop while (and
		   (next)
		   (< (+ r0
			 (* alpha (- (log alpha) (log (+ b w0)))))
		      t0)))

      (if (= a a0)
	  (/ w0 (+ b w0))
	  (/ b (+ b w0))))))



(defun !beta-bc (a0 a b)
  "Generates beta variances.

Algorithm: https://dl.acm.org/doi/pdf/10.1145/359460.359482

Note: !beta excepts that @c((min a b) <= 1)"
  (declare (optimize (speed 3) (safety 0) (debug 0))
	   (type single-float a0)
	   (type (single-float 0e0) a b))

  (unless (<= (min a b) 1.0)
    (error "cl-waffe:!beta failed because of (min a b) <= 1."))

  (let* ((alpha (+ a b))
  	 (beta  (/ b))
	 (gamma (+ 1 a (- b)))
	 (k1 (* gamma (+ 0.0138889 (* 0.0416667 b)) (/ (+ (* a b)
							  -0.777778))))
	 (k2 (+ 0.25 (* b (+ 0.5 (/ 0.258 gamma)))))
	 (z  0.0)
	 (y  0.0)
	 (v 0.0)
	 (w 0.0)
	 (f t)
	 (u1 0.0)
	 (u2 0.0))
    (declare (type single-float alpha beta gamma k1 k2 z y w v u1 u2))
    
    (labels ((next ()
	     (setq u1 (random 1.0))
	     (setq u2 (random 1.0))
	     (if (>= u1 0.5)
		 (progn
		   (setq z (* u1 u1 u2))
		   (if (<= z 0.25)
		       (progn
			 (setq v (* beta
				    (the single-float
					 (log (the (single-float 0e0)
						   (/ u1 (- 1 u1)))))))
			 (setq w (* a (exp v)))
			 nil)
		       (if (>= z k2)
			   t
			   nil)))
		 (progn
		   (setq y (* u1 u2))
		   (setq z (* u1 y))
		   (if (>= (+ (* 0.225 u2) z (- y))
			   k1)
		       t
		       nil)))))

      (loop while (and f (next))
	    do (progn
		 (setq v (* beta (log (the (single-float 0e0) (/ u1 (- 1 u1))))))
		 (setq w (* a (exp v)))

		 (if (< (- (* alpha
			      (log (the (single-float 0e0) (/ a (+ b w)))))
			   1.3862944)
			(log z))
		     (setq f nil))))

      (if (= a a0)
	  (/ w (+ b w))
	  (/ b (+ b w))))))

