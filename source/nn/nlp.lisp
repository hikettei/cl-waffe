
(in-package :cl-waffe.nn)

(defmodel RNNHiddenLayer (input-size
			  hidden-size
			  reccurent-weight
			  &key
			  (activation :tanh)
			  (bias nil)
			  (dropout nil))
  :parameters ((weight (parameter (!div (!randn `(,hidden-size ,input-size))
					(sqrt hidden-size))))
	       (reccurent-weight (if reccurent-weight
				     reccurent-weight
				     (!div (!randn `(,hidden-size ,hidden-size))
					   (sqrt hidden-size))))
               (bias (if bias
			 (parameter (!zeros `(,hidden-size 1)))
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
	      (setf (self reccurent-weight) h)
	      h1)))

(defmodel RNN (input-size
	       hidden-size
	       &key
	       (num-layers 1)
	       (activation :tanh)
	       (bias nil)
	       (dropout nil)
	       (biredical nil))
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
	       (h-w T))

  :forward ((x) ; X=(Batch_Size Sentence-Length Embedding-dim)
	    (if (eql (self h-w) T)
		(setf (self h-w)
		      (parameter (!zeros `(,(!shape x 2)
					   ,(self hidden-size))))))
	    (let ((h (!zeros `(,(self hidden-size)v1))))
	      (dotimes (i (self num-layers))
		(setq h (call (self rnn-layers) (const i) x h)))
	      h)))

;make !aref faster and 3d matmul

