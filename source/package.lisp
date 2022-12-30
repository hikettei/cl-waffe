
(in-package :cl-user)

(defpackage cl-waffe
  (:use :cl)
  (:export #:waffetensor
           #:tensor
	   #:const
	   #:data
	   #:grad
	   
	   #:defmodel
	   #:defnode
	   #:defoptimizer

	   #:forward
	   #:backward
	   #:parameters
	   #:hide-from-tree
	   
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
	   #:wf-tanh

	   #:print-model
	   
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
