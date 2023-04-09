
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
   #:*available-dtypes*)

  #| MPS APIs|#
  (:export
   #:matmul-mps))

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
	(if (second (find-config :mps plist))
	    (load-mps))))))

(defun load-mps ()
  (load-foreign-library "source/kernel_backends/mps/.build/release/libMPSBridge.dylib"))

; tmp

(defun backend-infomation (&optional (stream t))
  (format stream "Backend Information:"))
#|
(eval-when (:compile-toplevel :load-toplevel :execute)
  (load-configuration))

|#
