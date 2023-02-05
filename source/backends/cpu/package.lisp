
(in-package :cl-user)

(defpackage cl-waffe.backends.cpu
  (:documentation "An package for cpu kernel.")
  (:use :cl :cl-waffe)
  (:export #:dispatch-kernel #:infomation))
