
(in-package :cl-user)

(asdf:defsystem :cl-waffe
  :author "hikettei (Twitter:@icnhdm)"
  :licence "MIT"
  :version "0.1"
  :description "An deep learning framework for Common Lisp"
  :source-control (:git "https://github.com/hikettei/cl-waffe.git")
  :pathname "source"
  :depends-on (#:numcl
	       #:cl-ansi-text
	       #:mgl-mat
	       #:alexandria
	       #:cl-cuda
	       #:cl-libsvm-format
	       #:lparallel
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
			     (:file "matmul")
			     (:file "broadcast")
			     (:file "lazy-evaluate")			     
			     (:file "utils")
			     (:file "optimizers")			     
			     (:file "kernel")))
	       (:file "package")
	       (:file "utils")
	       (:file "conditions")
	       (:file "model" :depends-on ("tensor"))
	       (:file "dtype")
	       (:file "tensor")
	       (:file "distributions/random")
	       (:file "distributions/distribution")
	       (:file "distributions/beta")
	       (:file "distributions/gamma")
	       (:file "distributions/chisquare")
               (:file "kernel" :depends-on ("tensor"))
	       (:file "trainer" :depends-on ("optimizers"))
	       (:file "functions" :depends-on ("model"))
	       (:file "aref" :depends-on ("model"))
	       (:file "activations" :depends-on ("model"))
	       (:file "transpose" :depends-on ("model"))
	       (:file "operators" :depends-on ("tensor"))
	       (:file "sum" :depends-on ("model"))
	       (:file "conc" :depends-on ("model"))
	       (:file "iter" :depends-on ("model"))
	       (:file "shaping" :depends-on ("model"))
	       (:file "mathematicals" :depends-on ("model"))
	       (:file "einsum" :depends-on ("model"))
	       (:file "matrix-operations" :depends-on ("model"))
	       (:module "optimizers"
		:depends-on ("model")
		:components ((:file "package")
			     (:file "optimizers")
			     (:file "optimizer")))
	       (:module "nn"
		:depends-on ("model")
		:components ((:file "package")
			     (:file "weights")
			     (:file "utils")
			     (:file "losses")
			     (:file "attention")
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
					 (:file "node-extension")
					 (:file "copy")
					 (:file "caches")
					 (:file "nodes")
					 (:file "jit")
					 (:file "broadcast")
					 (:file "tensor-operate")
					 (:file "network")
					 (:file "operators")
					 (:file "optimizers"))))
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
						(:file "fnn")
						(:file "kftt-data-parser")))))

(asdf:defsystem :cl-waffe/benchmark
  :version nil
  :author "hikettei"
  :licence "MIT"
  :depends-on (:cl-waffe :cl-cram :cl-ppcre :clgplot)
  :components ((:module "benchmark"
		:components ((:file "package")
			     (:file "utils")
			     (:file "output")
			     (:file "benchmark")
			     (:file "benchmark1")))))
