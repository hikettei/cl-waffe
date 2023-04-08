
(in-package :cl-user)

(defpackage :cl-waffe.kernel
  (:use :cl :cl-waffe :cffi :alexandria)
  (:export
   ; Variables
   #:*dtype*
   #:*available-dtypes*))

(in-package :cl-waffe.kernel)

; Below is the configuration

(defparameter *backend* :cpu "The backend cl-waffe uses, in default: :mps (Mac's Metal, macOS only) cpu or cuda.")

; Todo: Make it settable

(defun load-blas ()
  (load-foreign-library "/usr/local/Cellar/openblas/0.3.22/lib/libblas.dylib"))

(defun load-cuda ()
  (format t "CUDA is unavailable"))

(defun load-onednn ()
  )

(defun load-mkl ()
  )

(defun backend-infomation (&optional (stream t))
  (format stream "Backend Information:"))

;(eval-when (:compile-toplevel :load-toplevel :execute)
    ;  (load-blas))

