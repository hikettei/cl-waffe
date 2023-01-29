
(defpackage :cl-waffe.backends.mgl
  (:use :cl :cl-waffe :mgl-mat :cl-cuda :cl-waffe.caches)
  (:export #:dispatch-kernel
	   #:adam-update
	   #:write-to-nth-dim-with-range
	   #:write-to-nth-dim-with-range1
	   #:create-thread-idx
	   #:get-difference
	   #:receive-delay
	   #:abort-delay))
