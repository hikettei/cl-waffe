
(in-package :cl-user)

#|
   Here's just some wrapper of scriba.
   For example: Automatically generates example.
|#
(defpackage :cl-waffe.documents
  (:use :cl :cl-waffe :cl-ppcre)
  (:export #:generate))

(in-package :cl-waffe.documents)

(defun write-scr (filepath content)
  (with-open-file (str filepath
                       :direction :output
                       :if-exists :supersede
                       :if-does-not-exist :create)
    (format str "~a" content)))


(defparameter *overview* "")
(defparameter *tutorials* "")
(defparameter *tips* "")
(defparameter *features-doc* "")

(defun generate ()
  (write-scr "./docs/overview.scr" *overview*)
  
  )

