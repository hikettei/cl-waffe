
(in-package :cl-user)

(defpackage cl-waffe.nn
  (:use :cl :cl-waffe)
  (:export :linear
           :linearlayer
	   :denselayer
	   
           :dropout
           :batchnorm2d
	   
	   :mse
	   :softmax-cross-entropy
           :cross-entropy))
