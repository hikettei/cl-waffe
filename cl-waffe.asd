
(in-package :cl-user)


(asdf:defsystem :cl-waffe
  :author "hikettei twitter -> @ichndm"
  :licence nil
  :version nil
  :description "an opencl-based deeplearning library"
  :pathname "source"
  :depends-on (#:numcl #:cl-ansi-text #:mgl-mat #:cl-libsvm-format #:alexandria #:sb-sprof #:cl-cuda)
  :in-order-to ((test-op (test-op cl-waffe-test)))
  :components ((:file "cl-cram")
	       (:module "cl-termgraph"
		:components ((:file "package")
			     (:file "cl-termgraph")
			     (:file "figure")))
	       (:module "backends/cpu"
		:depends-on ("package")
		:components ((:file "package")
			     (:file "kernel")))
	       (:module "backends/mgl-mat"
		:depends-on ("package" "tensor")
		:components ((:file "package")
			     (:file "kernel")))
	       (:file "package")
	       (:file "model" :depends-on ("tensor"))
	       (:file "tensor")
               (:file "kernel" :depends-on ("tensor"))
	       (:file "trainer" :depends-on ("optimizers"))
	       (:file "functions")
	       (:file "operators" :depends-on ("tensor"))
	       (:module "optimizers"
		:depends-on ("model")
		:components ((:file "package")
			     (:file "optimizers")
			     (:file "optimizer")))
	       (:module "nn"
		:components ((:file "package")
			     (:file "utils")
			     (:file "losses")
			     (:file "functional")))))
