
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

(defparameter *cl-waffe-doc* "")
(defparameter *nn-doc* "")
(defparameter *optimizers-doc* "")
(defparameter *conditions-doc* "")

(defparameter *apis-cl-waffe* "")
(defparameter *apis-cl-waffe-nn* "")
(defparameter *apis-cl-waffe-optimizers* "")


(defun generate ()
  (write-scr "./docs/overview.scr" *overview*)
  (write-scr "./docs/tutorial.scr" *tutorials*)
  (write-scr "./docs/tips.scr" *tips*)
  (write-scr "./docs/features.scr" *features-doc*)

  (write-scr "./docs/cl-waffe-doc.scr" *cl-waffe-doc*)
  (write-scr "./docs/nn-doc.scr" *nn-doc*)
  (write-scr "./docs/optimizers.scr" *optimizers-doc*)
  (write-scr "./docs/conditions.scr" *conditions-doc*)

  (write-scr "./docs/APIs_cl-waffe.scr" *apis-cl-waffe*)
  (write-scr "./docs/APIs_cl-waffe-nn.scr" *apis-cl-waffe-nn*)
  (write-scr "./docs/APIs_cl-waffe-optimizers.scr" *apis-cl-waffe-optimizers*)
  )

