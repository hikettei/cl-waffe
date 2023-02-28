(in-package :cl-user)

(asdf:defsystem :cl-waffe-benchmark
  :version nil
  :author "hikettei"
  :licence "MIT"
  :depends-on (:cl-waffe :cl-cram :cl-ppcre)
  :components ((:module "benchmark"
		:components ((:file "package")
			     (:file "utils")
			     (:file "output")
			     (:file "benchmark")))))

