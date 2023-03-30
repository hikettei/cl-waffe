
(in-package :cl-waffe)

(define-with-typevar !beta u (dims alpha beta)
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
	   (type u alpha beta)
	   (inline !beta-bb !beta-bc))
  (let* ((alpha (coerce alpha (quote u)))
	 (beta  (coerce beta  (quote u)))
	 (a (min alpha beta))
 	 (b (max alpha beta))
	 (result (!zeros dims))
	 (size (!size result)))
    (declare (type fixnum size))
    (with-facet (array ((data result) 'backing-array :direction :output))
      (declare (type (simple-array u) array))
      ; Todo For GPU.
      (loop for i fixnum upfrom 0 below size
	    do (setf (aref array i)
		     (if (>= a 1.0) ; Bug: (!beta ~ 1.0 1.0)
			 (!beta-bb alpha a b)
			 (!beta-bc alpha a b)))))
    result))


(declaim (ftype (function
		 (single-float
		  single-float
		  single-float)
		 single-float)
		!beta-bb-f
		!beta-bc-f))
(declaim (ftype (function
		 (double-float
		  double-float
		  double-float)
		 double-float)
		!beta-bb-d
		!beta-bc-d))
(define-with-typevar !beta-bb u (a0 a b)
  "Generates beta variances.

Algorithm: https://dl.acm.org/doi/pdf/10.1145/359460.359482

Note: !beta excepts that @c((min a b) > 1)"
  (declare (optimize (speed 3) (safety 0))
	   (type u a0)
	   (type (u 0e0) a b))

  (unless (>= (min a b) 1.0)
    (error "cl-waffe:!beta failed because of (min a b) > 1."))

  (let* ((alpha (+ a b))
  	 (beta  (sqrt (the (u 0e0)
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
	       (declare (type u u1 u2 v))
	       
	       (setq w0 (* a (exp v)))
	       (setq r0 (- (* gamma v) 1.3862944))
	       
	       (let* ((z (* u1 u1 u2))
		      (s (+ a r0 (- w0))))
		 (declare (type u z s))
		 
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


(define-with-typevar !beta-bc utype (a0 a b)
  "Generates beta variances.

Algorithm: https://dl.acm.org/doi/pdf/10.1145/359460.359482

Note: !beta excepts that @c((min a b) <= 1)"
  (declare (optimize (speed 3) (safety 0))
	   (type utype a0)
	   (type (utype 0e0) a b))

  (unless (<= (min a b) 1.0)
    (error "cl-waffe:!beta failed because of (min a b) <= 1."))

  (let* ((alpha (+ a b))
  	 (beta  (/ b))
	 (gamma (+ 1 a (- b)))
	 (k1 (* gamma (+ 0.0138889 (* 0.0416667 b)) (/ (+ (* a beta) -0.777778))))
	 (k2 (+ 0.25 (* b (+ 0.5 (/ 0.258 gamma)))))
	 (z  0.0)
	 (y  0.0)
	 (v 0.0)
	 (w 0.0)
	 (f t)
	 (u1 0.0)
	 (u2 0.0)
	 (lp t))
    (declare (type utype alpha beta gamma k1 k2 z y w v u1 u2)
	     (type boolean lp f))
    
    (labels ((next ()
	       (setq lp t)
	       (setq u1 (random 1.0))
	       (setq u2 (random 1.0))
	       (if (>= u1 0.5)
		   (progn
		     (setq z (* u1 u1 u2))
		     (if (<= z 0.25)
			 (progn
			   (setq v (* beta
				      (the utype
					   (- (log u1) (log (1+ (- u1)))))))
			   (setq w (* a (exp v)))
			   nil)
			 (if (>= z k2)
			     (progn
			       (setq lp nil)
			       t)
			     t)))
		   (progn
		     (setq y (* u1 u2))
		     (setq z (* u1 y))
		     (if (>= (+ (* 0.25 u2) z (- y)) k1)
			 (progn
			   (setq lp nil)
			   t)
			 t)))))

      (loop while (and f (next))
	    do (when lp
		 (setq v (* beta
			    (the utype
				 (- (log u1) (log (1+ (- u1)))))))
		 (setq w (* a (exp v)))
		 (if (>= (- (* alpha
			       (+ v
				  (log alpha)
				  (- (log (1+ (+ b w))))))
			    1.3862944)
			 (log z))
		     (setq f nil))))

      (if (= a a0)
	  (/ w (+ b w))
	  (/ b (+ b w))))))

