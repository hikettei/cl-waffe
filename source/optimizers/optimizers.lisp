
(in-package :cl-waffe.optimizers)

(defoptimizer SGD (params &key (lr 1e-3))
  :optimize t
  :parameters ((params params :type hash-table)
	       (lr lr :type single-float))
  :update (()
	   (dotimes (i (hash-table-count (self params)))
	     ; W(n+1) = W(n) - n * grad
             (!modify (gethash i (self params)) :-=
		      (!modify (grad (gethash i (self params))) :*= (self lr))))))

; not optimized
(defoptimizer Momentum (params &key (momentum 0.9) (lr 1e-3))
  :parameters ((params params) (lr 1e-2) (momentum momentum) (velocities (make-hash-table)))
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
  :parameters ((params params) (lr lr) (h (make-hash-table)) (epsilon epsilon) (decay-rate 0.99))
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
  
; still too slow...
(defoptimizer Adam (params &key (lr 1e-3) (epsilon 1e-7) (beta1 0.9) (beta2 0.999))
  :optimize t
  :parameters ((params params  :type hash-table)
	       (lr lr          :type single-float)
	       (m (make-hash-table) :type hash-table)
	       (v (make-hash-table) :type hash-table)
	       (n 0             :type fixnum)
	       (epsilon epsilon :type single-float)
	       (beta1 beta1 :type single-float)
	       (beta2 beta2 :type single-float))
  :update (()
	   (if (= (hash-table-count (self m)) 0)
	       (dotimes (i (hash-table-count (self params)))
		 (setf (gethash i (self m)) (data (!zeros (!shape (gethash i (self params))))))
		 (setf (gethash i (self v)) (data (!zeros (!shape (gethash i (self params))))))))
	   (incf (self n) 1)
	   (let ((lr-t (* (self lr) (/ (sqrt (- 1.0 (expt (self beta2) (self n))))
					     (- 1.0 (expt (self beta1) (self n)))))))
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
