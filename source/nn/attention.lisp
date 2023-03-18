
(in-package :cl-waffe.nn)

(defmodel ScaleDotProductAttention (n-dim)
  :optimize t
  :parameters ((n-dim n-dim :type fixnum))
  :forward ((q k v &optional (mask (const NIL)))
	    (let* ((scalar (sqrt (self n-dim)))
		   (attention-weight (!div
				      (!matmul q (!transpose k))
				      scalar)))
	      (when (data mask)
		(unless (eql
			 (the list (!shape mask))
			 (the list (!shape attention-weight)))
		  (error "Mismatched dim (todo more conds)"))

		(setq attention-weight (!mul attention-weight mask)))
	      (!matmul (!softmax attention-weight) v))))

; Todo: Optimize it with FlashAttention
(defmodel MultiHeadAttention (embedding-dim
			      num-heads
			      &key
			      (dropout-rate 0.0))
  :parameters ((embedding-dim embedding-dim :type fixnum)
	       (num-heads num-heads :type fixnum)
	       (dropout (if (= dropout-rate 0.0)
			    nil
			    (dropout dropout-rate)))
	       (head-dim (coerce (/ embedding-dim num-heads) 'fixnum)) ; Todo: Explict that embedding-dim/num-heads must be fixnum
	       (lk (linearlayer embedding-dim head-dim))
	       (lq (linearlayer embedding-dim head-dim))
	       (lv (linearlayer embedding-dim head-dim))
	       (l-out (linearlayer embedding-dim embedding-dim))
	       (attn (ScaleDotProductAttention embedding-dim)))
  :forward ((q k v &optional (mask (const NIL)))
	    (let ((batch-size (!shape q 0))
		  (seq-len    (!shape q 1))
		  (head-dim (self head-dim))
		  (num-heads (self num-heads)))
	      (let* ((q (!repeats q 0 num-heads))
		     (k (!repeats k 0 num-heads))
		     (v (!repeats v 0 num-heads))
		     (q (call (self lk) q))
		     (k (call (self lk) k))
		     (v (call (self lv) v))
		     (q (progn (!allow-destruct q) q))
		     (k (progn (!allow-destruct k) k))
		     (v (progn (!allow-destruct v) v))
		     (q (!reshape q
				  `(,(* num-heads batch-size)
				    ,seq-len
				    ,head-dim)))
		     (k (!reshape k
				  `(,(* num-heads batch-size)
				    ,seq-len
				    ,head-dim)))
		     (v (!reshape v
				  `(,(* num-heads batch-size)
				    ,seq-len
				    ,head-dim))))

		(when (data mask)
		  (setq mask (!repeats mask 0 num-heads)))

		(let ((chunks (!split
			       (call (self attn) q k v mask)
			       batch-size
			       :axis 0)))
		  (call (self l-out) (apply #'!concatenate 2 chunks)))))))
	       
			      
