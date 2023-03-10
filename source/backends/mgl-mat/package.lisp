
(defpackage :cl-waffe.backends.mgl
  (:documentation "An package for mgl-mat")
  (:use :cl :cl-waffe :mgl-mat :cl-cuda :cl-waffe.caches)
  (:export #:dispatch-kernel
	   #:adam-update
	   #:*use-blas-min-size
	   #:write-to-nth-dim-with-range
	   #:write-to-nth-dim-with-range1
	   #:create-thread-idx
	   #:get-difference
	   #:receive-delay
	   #:abort-delay
	   #:copy-elements
	   #:*verbose*
	   #:reset-jit
	   #:*force-lazy-eval*
	   #:*static-node-mode*
	   #:compile-and-run-lazy))

(setf mgl-mat:*default-lisp-kernel-declarations* `((optimize (speed 3) (safety 0) (compilation-speed 0))))
