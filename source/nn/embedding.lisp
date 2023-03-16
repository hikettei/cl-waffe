
(in-package :cl-waffe.nn)

(defnode EmbeddingTensor (padding-idx)
  :parameters ((xi T)
	       (weights T)
	       (padding-idx padding-idx))
  :forward ((x weights)
	    (save-for-backward xi x)
	    (save-for-backward weights weights)
	    (with-searching-calc-node :embedding-forward
	      x
	      weights
	      (self padding-idx)))
  :backward ((dy)
	     (let ((dx (with-searching-calc-node :embedding-backward
			 (self xi)
			 dy
			 (self weights)
			 (self padding-idx))))
	       (list dx ; x is supposed to be const, so usually not used.
		     dx))))

(defmodel Embedding (vocab-size
		     embedding-dim
		     &key
		     (pad-idx nil))
  :document (with-usage "Embedding"
	      :overview "Embedding"
	      :args "vocab-size embedding-dim &key (pad-idx nil)"
	      :forward "Emm"
	      :step-args "x")
  :parameters ((vocab-size vocab-size :type fixnum)
	       (embedding-dim embedding-dim :type fixnum)
	       (padding-idx (if pad-idx
				(const pad-idx)
				(const -1))
			    :type waffetensor)
	       (weights (parameter (!mul 0.01 (!randn `(,vocab-size, embedding-dim))))))
  :forward ((x)
	    "Embedding(x) where x is the shape of (batch-size length)"
	    (call (EmbeddingTensor (self padding-idx))
		  x
		  (self weights))))

