

(in-package :cl-user)

(asdf:defsystem :cl-waffe-examples
  :version nil
  :author "hikettei"
  :licence nil
  :depends-on (:cl-waffe :cl-libsvm-format :sb-sprof)
  :components ((:module "examples" :components ((:file "mnist")))))

