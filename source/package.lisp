
(in-package :cl-user)

(defpackage cl-waffe
  (:use :cl)
  (:export #:tensor
	   #:const
	   #:data
	   #:defmodel
	   #:call
	   #:backward
	   #:add
	   #:mul))
