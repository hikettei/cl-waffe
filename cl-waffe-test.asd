
(in-package :cl-user)

(defpackage :cl-waffe-test-asd
  (:use :cl :asdf :uiop))

(in-package :cl-waffe-test-asd)

(defsystem :cl-waffe-test
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
