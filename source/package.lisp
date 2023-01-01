
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
	   #:deftrainer
	   #:step-model

	   #:defdataset

	   #:model
	   #:update
	   #:zero-grad

	   #:forward
	   #:backward
	   #:parameters
	   #:hide-from-tree

	   #:train
	   #:get-dataset
	   #:get-dataset-length
	   
	   #:parameter
	   #:call
	   #:backward
	   #:zeros
	   #:randn
	   #:random-tensor
	   #:normal
	   #:arange
	   #:ones-like
	   #:array-ref
	   #:array-ref-expand
	   
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
