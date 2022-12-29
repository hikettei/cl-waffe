
(in-package :cl-user)

(defpackage :cl-waffe-test-asd
  (:use :cl :asdf :uiop))

(in-package :cl-waffe-test-asd)

(defsystem :cl-waffe-test
  :version nil
  :author "hikettei"
  :licence "MIT"
  :depends-on (:cl-waffe :fiveam)
  :components ((:module "t" :components ((:file "package")
					 (:file "operators")
					 (:file "network")
					 (:file "tensor-operate")
					 (:file "deriv"))))
  :perform (test-op (o s)
		    (symbol-call :fiveam :run! :test)))
