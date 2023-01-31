
(in-package :cl-user)

(defpackage cl-waffe.optimizers
  (:use :cl :cl-waffe)
  (:export
   :is-waffe-model
   :find-parameters
   :init-optimizer
   

   :SGD
   :Momentum
   :AdaGrad
   :RMSProp
   :Adam
   :RAdam
   :Amos))


