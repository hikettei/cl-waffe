
(in-package :cl-waffe.nn)


(defmodel RNNHiddenLayer (input-size
			  hidden-size
			  reccurent-weight
			  &key
			  (activation :relu)
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
	       (activation :relu)
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
	       (num-layers num-layers :type fixnum)
	       (hidden-size hidden-size :type fixnum)
	       (biredical biredical :type boolean)
	       (wo (linearlayer hidden-size hidden-size) :type linearlayer))
  :forward ((x &optional (hs (const nil)))
	    "Input: X = (BatchSize SentenceLength Embedding_Dim)
             Output (values x{t+1} h{t+1})"

	    (let* ((batch-size (!shape x 0))
		   (sentence-length (!shape x 1))
		   (hs-specified? (not (null (data hs))))
	      (if hs-specified?
		  (dotimes (w-i (length words))
		    (dotimes (rnn-i (self num-layers))
		      (setq hs (setf (!aref hs t w-i)
			    (call (self rnn-layers)
				  (const rnn-i)
				  (nth w-i words)
				  (!aref hs t w-i)))))
		    (setf (nth w-i words) (!aref hs t w-i)))
		  (dotimes (w-i (length words))
		    (dotimes (rnn-i (self num-layers))
		      (setq hs (call (self rnn-layers)
				     (const rnn-i)
				     (nth w-i words)
				     hs)))
		    (setf (nth w-i words) (!add 0.0 hs)))) ; this `!add` is intended to make a copy.
	      (call (self wo) (apply #'!concatenate 1 words))))))
