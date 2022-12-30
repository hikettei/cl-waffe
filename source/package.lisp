
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
	   #:ones-like
	   #:self

	   #:relu
	   #:sigmoid
	   #:tanh
	   
	   #:add
	   #:sub
	   #:mul
	   #:div
	   #:dot
	   #:pow
	   #:loge
	   #:t-exp
	   #:sum
	   #:mean
	   #:shape
	   #:reshape
	   #:repeats
	   #:transpose
	   #:astensor
	   #:matmul))
