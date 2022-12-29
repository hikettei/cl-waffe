
(in-package :cl-user)

(defpackage cl-waffe
  (:use :cl)
  (:export #:tensor
	   #:const
	   #:data
	   #:grad
	   #:defmodel
	   #:parameter
	   #:call
	   #:backward
	   #:zeros
	   #:randn
	   #:random-tensor
	   #:normal
	   #:arange
	   
	   #:add
	   #:sub
	   #:mul
	   #:div
	   #:dot
	   #:pow
	   #:loge
	   #:sum
	   #:mean
	   #:shape
	   #:reshape
	   #:repeats
	   #:transpose
	   #:astensor
	   #:matmul))
