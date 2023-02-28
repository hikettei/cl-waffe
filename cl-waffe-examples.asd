

(in-package :cl-user)

(asdf:defsystem :cl-waffe-examples
  :version nil
  :author "hikettei"
  :licence "MIT"
  :depends-on (:cl-waffe)
  :components ((:module "examples" :components ((:file "mnist")
						(:file "rnn"
						       :depends-on ("kftt-data-parser"))
						(:file "kftt-data-parser")))))

