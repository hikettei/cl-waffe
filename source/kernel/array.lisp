
(in-package :cl-waffe.kernel)

#|
  simple-array <-> BLAS/CUBLASの通信にFocus
  View Objectを効率的に扱えるように
|#

(define-with-typevar alloc-cpu-mat u (dims &key (initial-element 0))
  (let ((initial-element (coerce initial-element (quote u))))
    (foreign-alloc (dtypecase
		    (:short :short)
		    (:float :float)
		    (:double :double)
		    (T (error "Unavailbe dtype (add cases for me)")))
		   :initial-contents
		   (loop for i fixnum upfrom 0 below (apply #'* dims) collect initial-element))))

#|
(define-with-typevar alloc-cuda-mat u (dims &key (initial-element 0))
  (error "Unavailable"))
|#

(defun alloc-mat (dims &key (initial-element 0))
  (case *backend*
    (:cpu (alloc-cpu-mat dims
			 :initial-element initial-element))
    (T
     (error "Currently cl-waffe doesn't supports ~a" *backend*))))
