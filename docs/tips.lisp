
(in-package :cl-waffe.documents)

(defparameter *tips* "")

(with-page *tips* "Tips"
  (with-section "Destructive Operations")
  (with-section "Using other libraries with Facet APIs")
  (with-section "Zero-cost Transpose")
  (with-section "Logging"
    (insert "about: env vars, with-verbose")))
