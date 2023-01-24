
(in-package :cl-user)

(defpackage cl-waffe.optimizers
  (:use :cl :cl-waffe :sb-sprof)
  (:export
   :is-waffe-model
   :find-parameters
   ;:find-variables
   :init-optimizer
   

   :SGD
   :Momentum
   :AdaGrad
   :RMSProp
   :Adam
   :RAdam
   :Amos))


