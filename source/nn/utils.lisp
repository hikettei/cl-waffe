
(in-package :cl-waffe.nn)

; An implementation of Inverted Dropout.
(defnode Dropout (&optional (dropout-rate 0.5))
  :document (with-usage "Dropout"
	      :note "Todo: docstring")
  :parameters ((dropout-rate
		(if (and (> dropout-rate 0.0)
			 (< dropout-rate 1.0))
		    dropout-rate
		    (error "cl-waffe.nn: Dropout(x), x must be in the range of 0.0<x<1.0 where x is a single-float.")))
	       (mask T))
  :forward ((x)
	    (setf (self mask) (!zeros (!shape x)))
	    (if *no-grad* ; predict mode
		x
		(progn
		  (!modify (self mask) :bernoulli (self dropout-rate))
		  (!modify (!mul (self mask) x) :*= (/ 1 (- 1 (the single-float (self dropout-rate))))))))

  :backward ((dy)
	     (list (!mul (self mask) dy))))
