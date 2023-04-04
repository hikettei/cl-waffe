
(in-package :cl-user)

(defpackage cl-waffe.nn
  (:use :cl :cl-waffe)
  (:documentation "An packages for nn utils.")
  (:export
   :ScaleDotProductAttention
   :MultiHeadAttention)
  (:export
   :*weight-initializer*
   :init-weights
   :select-initializer
   :init-activation-weights)
  
  (:export :linear
           :linearlayer
	   :denselayer)
  
  (:export :RNN
	   :LSTM)

  (:export :dropout
           :batchnorm2d
           :embedding
	   :mse
	   :softmax-cross-entropy
           :cross-entropy))
