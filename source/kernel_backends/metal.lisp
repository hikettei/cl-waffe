
(in-package :cl-waffe.kernel)

; Todo: 3D gemm, ND gemm

(defcfun "mps_2dfgemm" :int
	  (alpha :double)
	  (a (:pointer :float))
	  (b (:pointer :float))
	  (beta :double)
	  (c (:pointer :float))
	  (m :int)
	  (n :int)
	  (k :int)
  (transpose_a :boolean)
  (transpose_b :boolean))

(define-with-typevar matmul-mps u (x y out
				   &key
				   (transpose-a nil)
				   (transpose-b nil))
					; To Add: case depending on dims, dtypes

  (with-facets ((x* ((data x) 'foreign-array :direction :input))
		(y* ((data y) 'foreign-array :direction :input))
		(o* ((data out) 'foreign-array :direction :input)))
    (let ((x* (slot-value x* 'mgl-mat::base-pointer))
	  (y* (slot-value y* 'mgl-mat::base-pointer))
	  (o* (slot-value o* 'mgl-mat::base-pointer)))
      (let ((m (!shape x 0))
	    (n (!shape y 1)) ; add assert k
	    (k (!shape x 1)))
	(mps-2dfgemm (coerce 1.0 'double-float)
		     x*
		     y*
		     (coerce 0.0 'double-float)
		     o*
		     m
		     n
		     k
		     transpose-a
		     transpose-b)))))

