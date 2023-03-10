
(in-package :cl-waffe.nn)

(defmodel RNNHiddenLayer (input-size
			  hidden-size
			  reccurent-weight
			  &key
			  (activation :tanh)
			  (bias nil)
			  (dropout nil))
  :parameters ((weight (parameter (!div (!randn `(,input-size
						  ,hidden-size))
					(sqrt hidden-size))))
	       (reccurent-weight (if reccurent-weight
				     reccurent-weight
				     (parameter (!div (!randn `(,hidden-size ,hidden-size))
						      (sqrt hidden-size)))))
               (bias (if bias
			 (parameter (!zeros `(1 ,hidden-size)))
			 nil))
	       (dropout (if dropout
			    (dropout dropout)
			    nil))
	       (activation (if (find activation `(:tanh :relu))
			       activation
			       (error "cl-waffe.nn.RNNHiddenLayer: available activations are following: :tanh :relu, but got ~a" activation))))

  :forward ((x h)
	    (let* ((h1 (!add (!matmul x (self weight))
			     (!matmul h (self reccurent-weight))))
		   (h1 (if (self bias)
			   (!add h1 (self bias))
			   h1))
		   (h1 (if (self dropout)
			   (call (self dropout) h1)
			   h1))
		   (h1 (case (self activation)
			   (:tanh (!tanh h1))
			   (:relu (!relu h1)))))
	      h1)))

(defmodel RNN (input-size
	       hidden-size
	       &key
	       (num-layers 1)
	       (activation :tanh)
	       (bias nil)
	       (dropout nil)
	       (biredical nil))
  :document (with-usage "RNN"
	      :overview "Todo: docs"
	      )
  :parameters ((rnn-layers (model-list
			    (loop for i upfrom 0 below num-layers
				  collect (RNNHiddenLayer
					   input-size
					   hidden-size
					   nil
					   :activation activation
					   :bias bias
					   :dropout dropout))))
	       (num-layers num-layers)
	       (hidden-size hidden-size)
	       (biredical biredical)
	       (wo (linearlayer hidden-size hidden-size)))
  :forward ((x &optional (hs (const nil)))
	    "Input: X = (BatchSize SentenceLength Embedding_Dim)
             Output (values x{t+1} h{t+1})"

	    (let* ((batch-size (!shape x 0))
		   (s-len (!shape x 1))
		   (hs (if (null (data hs))
			 (!zeros `(,batch-size
				   ,s-len
				   ,(self hidden-size)))
			 hs)))
	      
	      (if (self biredical)
		  ; when biredical=t, calc in the around way
		  (loop for xn downfrom (1- s-len) to 0
			do (let ((h (!zeros `(,batch-size
					      ,(self hidden-size))))
				 (xn-s (!squeeze (!aref x t xn t) 1)))
			     (dotimes (i (self num-layers))
			       (setq h (call (self rnn-layers)
					     (const i)
					     xn-s
					     h)))
			     (setq hs (setf (!aref hs t xn) h))))

		  ; when biredical=nil, calc in order.
		  (loop for xn upfrom 0 below s-len
			do (let ((h (!squeeze (!aref hs t xn t) 1))
				 (xn-s (!squeeze (!aref x t xn t) 1)))
			     (dotimes (i (self num-layers))
			       (setq h (call (self rnn-layers)
					     (const i)
					     xn-s
					     h)))
			     (setq hs (setf (!aref hs t xn) h)))))
	      (call (self wo) hs))))


