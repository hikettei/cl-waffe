
(in-package :cl-user)


(asdf:defsystem :cl-waffe
  :author "hikettei twitter -> @ichndm"
  :licence nil
  :version nil
  :description "an opencl-based deeplearning library"
  :pathname "source"
  :depends-on (#:numcl #:cl-ansi-text #:mgl-mat #:cl-libsvm-format)
  :in-order-to ((test-op (test-op cl-waffe-test)))
  :components ((:file "cl-cram")
	       (:module "cl-termgraph"
		:components ((:file "package")
			     (:file "cl-termgraph")
			     (:file "figure")))
	       (:module "backends/cpu"
		:components ((:file "package")
			     (:file "kernel")))
	       (:module "backends/mgl-mat"
		:components ((:file "package")
			     (:file "kernel")))
	       (:module "backends/opencl"
	       :components ((:file "package")
			    (:file "kernel")))
	       (:file "package" :depends-on ("backends/cpu"
					     "backends/opencl"))
	       (:file "model" :depends-on ("tensor"))
	       (:file "tensor")
	       (:file "kernel")
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
