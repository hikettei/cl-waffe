
(in-package :cl-waffe.nn)

(defmodel RNNHiddenLayer (input-size
			  hidden-size
			  &key
			  (activation :relu)
			  (bias nil)
			  (dropout nil))
  :parameters ((weight (parameter (!div (!randn `(,input-size
						  ,hidden-size))
					(sqrt hidden-size))))
	       (reccurent-weight (parameter
				  (!div (!randn `(,hidden-size ,hidden-size))
					(sqrt hidden-size))))
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
	       (bidirectional nil))
  :parameters ((rnn-layers (model-list
			    (loop for i upfrom 0 below num-layers
				  collect (RNNHiddenLayer
					   input-size
					   hidden-size
					   :activation activation
					   :bias bias
					   :dropout dropout))))
	       (num-layers num-layers :type fixnum)
	       (hidden-size hidden-size :type fixnum)
	       (biredical bidirectional :type boolean)
	       (wo (linearlayer hidden-size hidden-size) :type linearlayer))
  :forward ((x &optional (hs (const nil)))
	    "Input: X = (BatchSize SentenceLength Embedding_Dim)
             Output (values x{t+1} h{t+1})"

	    (let* ((batch-size (!shape x 0))
		   (hs-specified? (not (null (value hs))))
		   (hs (if (null (value hs))
			   (!zeros `(,batch-size
				     1
				     ,(self hidden-size)))
			   hs))
		   (words (!split x 1 :axis 1))
		   (hs1))

	      ; Nodes from x is correctly lazy-evaluated regardless of hs.

	      ; If bidirectional, W[n] -> W[n-1] -> ... -> W[1]
	      (unless (self biredical)
		(setq words (reverse words)))

	      (if hs-specified?
		  (dotimes (w-i (length words))
		    (setq hs1 (!aref hs t w-i t))
		    (dotimes (rnn-i (self num-layers))
		      (setq hs1 (call (self rnn-layers)
				      (const rnn-i)
				      (nth w-i words)
				      hs1)))
		    (setf (nth w-i words) hs1))
		  (dotimes (w-i (length words))
		    (dotimes (rnn-i (self num-layers))
		      (setq hs (call (self rnn-layers)
				     (const rnn-i)
				     (nth w-i words)
				     hs)))
		    (setf (nth w-i words) (!add 0.0 hs)))) ; This !add is intended to create a copy.
	      (call (self wo) (apply #'!concatenate 1 words)))))

