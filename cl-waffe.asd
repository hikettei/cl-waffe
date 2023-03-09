
(in-package :cl-user)

(ql:quickload :lparallel :silent t)

(asdf:defsystem :cl-waffe
  :author "hikettei (Twitter:@icnhdm)"
  :licence "MIT"
  :version "0.1"
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
	       (:file "distributions/random")
	       (:file "distributions/distribution")
	       (:file "distributions/beta")
	       (:file "distributions/gamma")
	       (:file "distributions/chisquare")
               (:file "kernel" :depends-on ("tensor"))
	       (:file "trainer" :depends-on ("optimizers"))
	       (:file "functions" :depends-on ("model"))
	       (:file "aref")
	       (:file "activations")
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


(defpackage :cl-waffe-test-asd
  (:use :cl :asdf :uiop))

(in-package :cl-waffe-test-asd)

(defsystem :cl-waffe/test
  :version nil
  :author "hikettei"
  :licence "MIT"
  :depends-on (:cl-waffe :fiveam :cl-libsvm-format)
  :components ((:module "t" :components ((:file "package")
					 (:file "utils")
					 (:file "deriv")
					 (:file "copy")
					 (:file "caches")
					 (:file "nodes")
					 (:file "jit")
					 (:file "broadcast")
					 (:file "tensor-operate")
					 (:file "network")
					 (:file "operators")
					 (:file "optimizers")
					 )))
  :perform (test-op (o s)
		    (symbol-call :fiveam :run! :test)))

(in-package :cl-user)

(asdf:defsystem :cl-waffe/examples
  :version nil
  :author "hikettei"
  :licence "MIT"
  :depends-on (:cl-waffe)
  :components ((:module "examples" :components ((:file "mnist")
						(:file "rnn"
						       :depends-on ("kftt-data-parser"))
						(:file "kftt-data-parser")))))

(asdf:defsystem :cl-waffe/benchmark
  :version nil
  :author "hikettei"
  :licence "MIT"
  :depends-on (:cl-waffe :cl-cram :cl-ppcre)
  :components ((:module "benchmark"
		:components ((:file "package")
			     (:file "utils")
			     (:file "output")
			     (:file "benchmark")))))
