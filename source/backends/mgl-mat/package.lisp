
(defpackage :cl-waffe.backends.mgl
  (:documentation "An package for mgl-mat")
  (:use :cl :cl-waffe :mgl-mat :cl-cuda :cl-waffe.caches)
  (:export #:dispatch-kernel
	   #:adam-update
	   #:write-to-nth-dim-with-range
	   #:write-to-nth-dim-with-range1
	   #:create-thread-idx
	   #:get-difference
	   #:receive-delay
	   #:abort-delay
	   #:copy-elements
	   #:*verbose*
	   #:compile-and-run-lazy))
