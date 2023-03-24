
(in-package :cl-user)

(defpackage :cl-waffe-benchmark
  (:use :cl :cl-waffe :mgl-mat :clgplot)
  (:export
   #:start-benchmark
   #:compare-to-python))
