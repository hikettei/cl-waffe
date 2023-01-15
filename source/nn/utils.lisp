
(in-package :cl-waffe.nn)

(defmodel LinearLayer (in-features out-features &optional (bias T))
  :optimize t
  :parameters ((weight (parameter (!mul 0.01 (!randn `(,in-features ,out-features)))))
	      (bias (if bias
			(parameter (!zeros `(,out-features 1)))
			nil)))
  :forward ((x) (cl-waffe.nn:linear x (cl-waffe:self weight) (cl-waffe:self bias))))


(defmodel DenseLayer (in-features out-features &optional (bias T) (activation :relu))
  :optimize t
  :parameters ((layer (linearlayer in-features out-features bias)) (activation activation))
  :forward ((x)
	    (case (cl-waffe:self activation)
	      (:relu
	       (!relu (call (cl-waffe:self layer) x)))
	      (:sigmoid
	       (!sigmoid (call (cl-waffe:self layer) x)))
	      (:tanh
	       (!tanh (call (cl-waffe:self layer) x)))
	      (:softmax
	       (!softmax (call (cl-waffe:self layer) x)))
	      (T
	       (funcall (cl-waffe:self activation)
			(call (cl-waffe:self layer) x))))))
    
; An implementation of Inverted Dropout.
(defnode Dropout (&optional (dropout-rate 0.5))
  :optimize nil
  :parameters ((dropout-rate
		(if (and (> dropout-rate 0.0)
			 (< dropout-rate 1.0))
		    dropout-rate
		    (error "cl-waffe.nn: Dropout(x), x must be in the range of 0.0<x<1.0 where x is a single-float."))
		:type
		single-float)
	       (mask T))
  :forward ((x)
	    (if (eql (self mask) T) ; is first call?
		(setf (self mask) (!zeros (!shape x))))
	    
	    (if *no-grad* ; predict mode
		x
		(progn
		  (!modify (self mask) :bernoulli (self dropout-rate))
		  (!modify (!mul (self mask) x) :*= (/ 1 (- 1 (self dropout-rate)))))))

  :backward ((dy)
	     (list (!mul (self mask) dy))))

