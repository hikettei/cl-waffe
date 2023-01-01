
(in-package #:cl-user)

(asdf:defsystem :cl-termgraph
  :name "cl-termgraph"
  :depends-on (:cl-ansi-text)
  :description "A graphic library that plots graph for Common Lisp"
  :author "hikettei"
  :license "MIT"
  :serial t
  :components ((:file "package")
	       (:file "cl-termgraph")
	       (:file "figure")))

