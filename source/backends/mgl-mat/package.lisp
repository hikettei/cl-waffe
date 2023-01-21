
(defpackage :cl-waffe.backends.mgl
  (:use :cl :cl-waffe :mgl-mat :cl-cuda)
  (:export #:dispatch-kernel
	   #:adam-update
	   #:write-to-nth-dim-with-range))
