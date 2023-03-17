
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
	      attention-weight)))

