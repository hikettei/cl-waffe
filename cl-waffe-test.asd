
(in-package :cl-user)

(defpackage :cl-waffe-test-asd
  (:use :cl :asdf :uiop))

(in-package :cl-waffe-test-asd)

(defsystem :cl-waffe-test
  :version nil
  :author "hikettei"
  :licence nil
  :depends-on (:cl-waffe :fiveam :cl-libsvm-format :sb-sprof)
  :components ((:module "t" :components ((:file "package")
					 ;(:file "mnist")
					 (:file "deriv")				
					 (:file "tensor-operate")
					 (:file "operators")
					 (:file "network")
					 ;(:file "deriv")
					 )))
  :perform (test-op (o s)
		    (symbol-call :fiveam :run! :test)))

