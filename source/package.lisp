
(in-package :cl-user)

(defpackage cl-waffe
  (:use :cl :mgl-mat :alexandria)
  (:export #:waffetensor
           #:tensor
	   #:const
	   #:sysconst
	   
	   #:data
	   #:grad

	   #:waffedatatype
	   #:waffe-array

	   #:with-no-grad
	   #:*no-grad*

	   #:waffe-tensor-p
	   #:waffetensor-is-next-destruct?
	   
	   #:defmodel
	   #:defnode
	   #:defoptimizer
	   #:deftrainer
	   #:defdataset

	   #:save-for-backward
	   
	   #:step-model
	   #:predict
	   #:get-dataset
	   #:get-dataset-length

	   #:with-optimized-operation

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

	   #:!set-batch
	   #:!reset-batch

	   #:waffetensor-destructively-calln
	   #:waffetensor-destructive?
	   #:waffetensor-is-data-destructed?
	   #:waffetensor-report-index
	   #:with-ignore-optimizer
	   #:*ignore-optimizer*

	   #:self

	   #:relu
	   #:sigmoid
	   #:wf-tanh

	   #:print-model

	   #:*default-backend*
	   #:extend-from
	   #:!zeros
	   #:!ones
	   #:!fill
	   #:!arange
	   #:!aref
	   #:!row-major-aref
	   #:!with-mgl-operation
	   #:!copy
	   #:!index
	   #:!where
	   #:!random
	   #:!random-with
	   #:!normal
	   #:!randn
	   #:!bernoulli
	   #:!beta
	   #:!gamma
	   #:!chisquare
	   #:!shape
	   #:!dims
	   #:!size
	   #:!size-1
	   #:!zeros-like
	   #:!ones-like
	   #:!full-like

	   #:with-calling-layers

	   #:!add
	   #:!sub
	   #:!mul
	   #:!div

	   #:!dot
	   #:!sum

	   #:!sqrt
	  
	   #:!pow
	   #:!mean
	   #:!log
	   #:!reshape
	   #:!transpose
	   #:!exp
	   #:!matmul
	   #:!repeats

	   #:!squeeze
	   #:!unsqueeze

	   #:!modify
	   
	   #:!relu
	   #:!sigmoid
	   #:!tanh
	   #:!softmax
	   ))
