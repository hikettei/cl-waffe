
(in-package :cl-user)

#|
Todo:
Extend mgl-mat:vec class and support :short-float,
Writing Metal Kernels.
|#

(defpackage :cl-waffe.kernel
  (:documentation "A kernel for cl-waffe, which enables MPS backend, FP16, view.")
  (:use :cl :cl-waffe :cffi :alexandria :mgl-mat :mgl-cube)
  (:export
   ; Variables
   #:*dtype*
   #:*available-dtypes*))

(in-package :cl-waffe.kernel)

(defparameter cl-user::*cl-waffe-configuration*
  `((:mps nil))) ; with-backend :mps de kidou

(defun find-config (name list)
  (loop for i fixnum upfrom 0 below (length list)
	if (eql (car (nth i list)) name)
	  do (return-from find-config (nth i list))))

(defun load-configuration ()
  (let* ((config 'cl-user::*cl-waffe-configuration*)
	 (plist (when (boundp config)
		  (symbol-value config))))
    
    (unless plist
      (warn "cl-user::*cl-waffe-configuration* is not found.")
      (progn
	(load-blas (find-config :BLAS plist))
	(if (find-config :CUDA plist)
	    (load-cuda))
	))))

(defun load-blas (path)
  (load-foreign-library path))

(defun load-cuda ()
  (format t "CUDA is unsupported by me currently. Consider using mgl-mat backend."))

(defun load-mps ()
  (load-foreign-library "source/kernel_backends/mps/.build/release/libMPSBridge.dylib"))

; tmp
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

(defun backend-infomation (&optional (stream t))
  (format stream "Backend Information:"))
#|
(eval-when (:compile-toplevel :load-toplevel :execute)
  (load-configuration))

|#
