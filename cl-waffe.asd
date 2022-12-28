
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
  ;:depends-on (#:numcl)
  :in-order-to ((test-op (test-op cl-waffe-test)))
  :components ((:module "backends/cpu"
		:components ((:file "package")
			     (:file "kernel")))
	       (:module "backends/opencl"
	       :components ((:file "package")
			    (:file "kernel")))
	       (:file "package" :depends-on ("backends/cpu"
					     "backends/opencl"))
	       (:file "kernel")
	       (:file "model")
	       (:file "tensor")
	       (:file "functions")
	       (:file "operators")))
