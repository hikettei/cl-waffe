
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
	       :static-vectors
	       #:mgl-mat
	       #:alexandria
	       #:cl-cuda
	       #:cffi
	       #:cl-libsvm-format
	       #:lparallel
	       #:trivial-garbage
	       #:bordeaux-threads)
  :serial t
  :in-order-to ((test-op (test-op cl-waffe-test)))
  :components ((:file "cl-cram")
	       (:file "package")
	       (:file "utils")
	       (:file "dtype")
	       (:file "conditions")
	       ; the below is to be deleted
	       (:module "backends/mgl-mat"
		:components ((:file "cache")
			     (:file "package")
			     (:file "matmul")
			     (:file "broadcast")
			     (:file "lazy-evaluate")			     
			     (:file "utils")
			     (:file "optimizers")			     
			     (:file "kernel")))
	       (:module "backends/cpu"
		:components ((:file "package")
			     (:file "kernel")))
	       (:module "kernel_backends"
		:components ((:file "package")
			     (:file "array")
			     (:file "cuda")
			     (:file "blas")))
	       
	       (:file "model")
	       (:file "tensor")
	       (:file "thread")
	       (:file "distributions/random")
	       (:file "distributions/distribution")
	       (:file "distributions/beta")
	       (:file "distributions/gamma")
	       (:file "distributions/chisquare")
               (:file "kernel")
	       (:file "functions")
	       (:file "aref")
	       (:file "activations")
	       (:file "transpose")
	       (:file "operators")
	       (:file "sum")
	       (:file "conc")
	       (:file "iter")
	       (:file "shaping")
	       (:file "mathematicals")
	       (:file "einsum")
	       (:file "matrix-operations")
	       (:module "optimizers"
		:depends-on ("model")
		:components ((:file "package")
			     (:file "optimizers")
			     (:file "optimizer")))
	       (:file "trainer")
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
	       (:module "impls/mps"
		:components ((:file "package")
			     (:file "mathematicals")))
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
  :serial t
  :components ((:module "t" :components ((:file "package")
					 (:file "utils")
					 (:file "deriv")
					 (:file "node-extension")
					 (:file "copy")
					 (:file "dtype")
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
  :depends-on (:cl-waffe :cl-cram :cl-ppcre :clgplot :shasht)
  :components ((:module "benchmark"
		:components ((:file "package")
			     (:file "utils")
			     (:file "output")
			     (:file "benchmark")
			     (:file "benchmark1")))))


(asdf:defsystem :cl-waffe/documents
  :version nil
  :author "hikettei"
  :licence "MIT"
  :depends-on (:cl-waffe :cl-ppcre)
  :components ((:module "docs"
		:components ((:file "package")
			     (:file "document")
			     (:file "overview")
			     (:file "tutorial")
			     (:file "tips")
			     (:file "features")
			     (:file "cl-waffe-docs")
			     (:file "cl-waffe-nn-docs")
			     (:file "cl-waffe-optimizers-doc")
			     (:file "conditions")
			     ))))
