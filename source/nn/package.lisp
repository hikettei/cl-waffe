
(in-package :cl-user)

(defpackage cl-waffe.nn
  (:use :cl :cl-waffe)
  (:documentation "An packages for nn utils.")
  (:export
   :ScaleDotProductAttention
   :MultiHeadAttention)
  (:export :linear
           :linearlayer
	   :denselayer)

  (:export
           :dropout
           :batchnorm2d

           :embedding
           :RNN
   
	   :mse
	   :softmax-cross-entropy
           :cross-entropy))
