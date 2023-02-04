
(in-package :cl-user)

(defpackage :cl-waffe.io
  (:documentation "A set of dataloader")
  (:use :cl :cl-libsvm-format :mgl-mat :cl-waffe)
  (:export #:read-libsvm-data))
