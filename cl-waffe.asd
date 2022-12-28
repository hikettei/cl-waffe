
(in-package :cl-user)

(defpackage :cl-waffe-asd
  (:use :cl :asdf))

(in-package :cl-waffe-asd)

(asdf:defsystem :cl-waffe
  :author "hikettei twitter -> @ichndm"
  :licence "MIT"
  :version nil
  :description "an opencl-based deeplearning library"
  :pathname "source"
  :in-order-to ((test-op (test-op cl-waffe-test)))
  :components ((:file "package")
	       (:file "model")
	       (:file "tensor")
	       (:file "functions")))
