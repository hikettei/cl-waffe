
(in-package :cl-waffe.optimizers)

(defoptimizer SGD (params &key (lr 1e-3))
  :document (with-usage "SGD"
	      :overview "Simple SGD."
	      :args "&key (lr 1e-3)"
	      :update "Following defnition.")
  :parameters ((params params :type hash-table)
	       (lr lr :type float))
  :update (()
	   (dotimes (i (hash-table-count (self params)))
	     ; W(n+1) = W(n) - n * grad
             (!modify (gethash i (self params)) :-=
		      (!mul (grad (gethash i (self params))) (self lr)))

	     )))

; not optimized
(defoptimizer Momentum (params &key (momentum 0.9) (lr 1e-3))
  :document (with-usage "Momentum"
	      :overview "Simple Momentum"
	      :note "This code isn't optimized and slow"
	      :args "&key (momentum 0.9) (lr 1e-3)"
	      :update "Following definition.")
  :parameters ((params params) (lr lr) (momentum momentum) (velocities (make-hash-table)))
  :update (()
	   (if (= (hash-table-count (self velocities)) 0)
	       (progn
		 (dotimes (i (hash-table-count (self params)))
		   (setf (gethash i (self velocities)) 0))))
	   (dotimes (i (hash-table-count (self params)))
	     ; v(n+1) = momentum*v(n) - grad*lr
	     ; w(n+1) = w(n) + v(n+1)

	     (setf (gethash i (self velocities)) (data (!sub (!mul (self momentum) (gethash i (self velocities)))
							     (!mul (self lr) (grad (gethash i (self params)))))))
	     (mgl-mat:copy! (data (!add (gethash i (self velocities))
					(gethash i (self params))))
			    (data (gethash i (self params)))))))

; not optimized
(defoptimizer AdaGrad (params &key (lr 1e-3) (epsilon 1e-7))
  :document (with-usage "AdaGrad"
	      :overview "Simple AdaGrad"
	      :note "The codes aren't optimized and slow. Todo: Write docs")
  :parameters ((params params) (lr lr) (h (make-hash-table)) (epsilon epsilon))
  :update (()
	   (if (= (hash-table-count (self h)) 0)
	       (dotimes (i (hash-table-count (self params)))
		 (setf (gethash i (self h)) 0)))
	   (dotimes (i (hash-table-count (self params)))
	     ; h(t+1) = h(t) + (grad * grad)
             ; w(t+1) = w(t) - {lr * grad}/sqrt(h(t+1))
	     (setf (gethash i (self h)) (data (!add (gethash i (self h))
						(!mul (grad (gethash i (self params)))
						      (grad (gethash i (self params)))))))
	     (mgl-mat:copy! (data (!sub (data (gethash i (self params)))
					(!div (!mul (self lr) (grad (gethash i (self params))))
					      (!add (!sqrt (gethash i (self h))) (self epsilon)))))
			    (data (gethash i (self params)))))))

; not optimized
(defoptimizer RMSProp (params &key (lr 1e-3) (epsilon 1e-7) (decay-rate 0.99))
  :document (with-usage "RMSProp"
	      :overview "Simple RMSProp"
	      :note "Not Optimized and slow, todo: write docs")
  :parameters ((params params) (lr lr) (h (make-hash-table)) (epsilon epsilon) (decay-rate decay-rate))
  :update (()
	   (if (= (hash-table-count (self h)) 0)
	       (dotimes (i (hash-table-count (self params)))
		 (setf (gethash i (self h)) 0)))
	   (dotimes (i (hash-table-count (self params)))
             (setf (gethash i (self h)) (data (!mul (gethash i (self h)) (self decay-rate))))
	     (setf (gethash i (self h)) (data (!add (gethash i (self h))
						(!mul (!mul (!sub 1.0 (self decay-rate))
							    (grad (gethash i (self params))))
						      (grad (gethash i (self params)))))))
	     (mgl-mat:copy! (data (!sub (data (gethash i (self params)))
					(!div (!mul (self lr) (grad (gethash i (self params))))
					      (!add (!sqrt (gethash i (self h))) (self epsilon)))))
			    (data (gethash i (self params)))))))


(defoptimizer Adam (params &key (lr 1e-3) (epsilon 1e-7) (beta1 0.9) (beta2 0.999))
  :document (with-usage "Adam"
	      :overview "Simple Adam. It invokes kernel directly."
	      :args "&key (lr 1e-3) (epsilon 1e-7) (beta1 0.9) (beta2 0.999)"
	      :update "Following definition.")
  :parameters ((params params :type hash-table)
	       (lr lr :type float)
	       (m (make-hash-table) :type hash-table)
	       (v (make-hash-table) :type hash-table)
	       (n 0 :type fixnum)
	       (epsilon epsilon :type float)
	       (beta1 beta1 :type float)
	       (beta2 beta2 :type float))
  :update (()
	   (if (= (hash-table-count (self m)) 0)
	       (dotimes (i (hash-table-count (self params)))
		 (setf (gethash i (self m)) (data (!zeros (!shape (gethash i (self params))))))
		 (setf (gethash i (self v)) (data (!zeros (!shape (gethash i (self params))))))))
	   (setf (self n) (+ (self n) 1))
	   (let ((lr-t (* (self lr) (/ (sqrt (the (single-float 0e0)
						  (- 1.0 (expt
							  (the
							   (single-float 0e0)
							   (self beta2))
						          (the fixnum
							       (self n))))))
				       (- 1.0
					  (the single-float
					       (expt
						(self beta1)
						(self n))))))))
	     (dotimes (i (hash-table-count (self params)))
	       (cl-waffe.backends.mgl:adam-update
		            (gethash i (self m))
			    (gethash i (self v))
			    (self beta1)
			    (self beta2)
			    (data (gethash i (self params)))
			    (grad (gethash i (self params)))
			    (mgl-mat:mat-size (gethash i (self m)))
			    (self epsilon)
			    lr-t)))))
