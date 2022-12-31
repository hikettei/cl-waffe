
(in-package :cl-user)

(defpackage :cl-waffe-asd
  (:use :cl :asdf))

(in-package :cl-waffe-asd)

(asdf:defsystem :cl-waffe
  :author "hikettei twitter -> @ichndm"
  :licence nil
  :version nil
  :description "an opencl-based deeplearning library"
  :pathname "source"
  :depends-on (#:numcl #:cl-ansi-text)
  :in-order-to ((test-op (test-op cl-waffe-test)))
  :components ((:file "cl-cram")
	       (:module "cl-termgraph"
		:components ((:file "package")
			     (:file "cl-termgraph")
			     (:file "figure")))
	       (:module "backends/cpu"
		:components ((:file "package")
			     (:file "kernel")))
	       (:module "backends/opencl"
	       :components ((:file "package")
			    (:file "kernel")))
	       (:file "tensor" :depends-on ("package"))
	       (:file "package" :depends-on ("backends/cpu"
					     "backends/opencl"))
	       (:file "kernel")
	       (:file "model")
	       (:file "trainer" :depends-on ("optimizers"))
	       (:file "functions")
	       (:file "operators")
	       (:module "optimizers"
		:components ((:file "package")
			     (:file "optimizers")
			     (:file "optimizer")))
	       (:module "nn"
		:components ((:file "package")
			     (:file "utils")
			     (:file "losses")
			     (:file "functional")))))
