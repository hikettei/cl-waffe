
(in-package :cl-waffe.nn)

; not working...
(defnode Embedding (vocab-size
		    embedding-dim
		    &optional
		    (pad-idx nil))
  :parameters ((xi T)
	       (vocab-size vocab-size :type fixnum)
	       (embedding-dim embedding-dim :type fixnum)
	       (padding-idx (if padding-idx
				padding-idx
				-1)
			    :type fixnum)
	       (weights (parameter (!mul 0.01 (!randn `(,vocab-size, embedding-dim))))))

  :forward ((x)
	    "Embedding(x) where x is the shape of (batch-size length)"
	    (setf (self xi) x)
	    (with-searching-calc-node :embedding-forward
	      x
	      (self weights)
	      (self vocab-size)
	      (self padding-idx)))

  :backward ((dy)
	     (list (with-searching-calc-node :embedding-backward (self x) (self weights)))))
