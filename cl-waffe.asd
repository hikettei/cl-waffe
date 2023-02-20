
(in-package :cl-user)

(ql:quickload :lparallel :silent t)

(asdf:defsystem :cl-waffe
  :author "hikettei (Twitter:@icnhdm)"
  :licence "MIT"
  :version nil
  :description "An deep learning framework for Common Lisp"
  :source-control (:git "https://github.com/hikettei/cl-waffe.git")
  :pathname "source"
  :depends-on (#:numcl
	       #:lake
	       #:cl-ansi-text
	       #:mgl-mat
	       #:alexandria
	       #:cl-cuda
	       #:cl-libsvm-format
	       #:inlined-generic-function
	       #:trivial-garbage
	       #:bordeaux-threads)
  :in-order-to ((test-op (test-op cl-waffe-test)))
  :components ((:file "cl-cram")
	       (:module "cl-termgraph"
		:components ((:file "package")
			     (:file "cl-termgraph")
			     (:file "figure")))
	       (:module "backends/cpu"
		:depends-on ("package"
			     "backends/mgl-mat")
		:components ((:file "package")
			     (:file "kernel")))
	       (:module "backends/mgl-mat"
		:depends-on ("package" "tensor")
		:components ((:file "cache")
			     (:file "package")
			     (:file "lazy-evaluate")			     
			     (:file "utils")
			     (:file "optimizers")			     
			     (:file "kernel")))
	       (:file "package")
	       (:file "utils")
	       (:file "model" :depends-on ("tensor"))
	       (:file "tensor")
               (:file "kernel" :depends-on ("tensor"))
	       (:file "trainer" :depends-on ("optimizers"))
	       (:file "functions" :depends-on ("model"))
	       (:file "operators" :depends-on ("tensor"))
	       (:module "optimizers"
		:depends-on ("model")
		:components ((:file "package")
			     (:file "optimizers")
			     (:file "optimizer")))
	       (:module "nn"
		:depends-on ("model")
		:components ((:file "package")
			     (:file "utils")
			     (:file "losses")
			     (:file "functional")
			     (:file "norms")
			     (:file "nlp")
			     (:file "layers")
			     (:file "embedding")
			     (:file "cnn")))
	       (:module "io"
		:components ((:file "package")
			     (:file "libsvm")))))
