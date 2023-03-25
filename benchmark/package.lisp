
(in-package :cl-user)

(defpackage :cl-waffe-benchmark
  (:use :cl :cl-waffe :mgl-mat :clgplot :shasht)
  (:export
   #:start-benchmark
   #:compare-to-python
   #:generate-result))
