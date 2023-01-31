
(in-package :cl-user)

(defpackage cl-waffe
  (:documentation "An package for defining node, initializing and computing with tensor, and backprops.")
  (:use :cl :mgl-mat :alexandria)
  (:export
           ; Functions and structures for tensor
           #:waffetensor
           #:tensor
	   #:const
	   #:sysconst

	   ; An parameters for displaying tensor.
	   #:*print-char-max-len*
	   #:*print-arr-max-size*
	   #:*print-mat-max-size*

	   ; Functions for using tensor's data
	   #:warranty
	   #:data
	   #:grad

	   #:waffedatatype
	   #:waffe-array

	   #:waffetensor-thread-data

	   #:with-no-grad
	   #:*no-grad*

	   #:waffe-tensor-p
	   #:waffetensor-is-next-destruct?

	   #:with-searching-calc-node
	   #:defmodel
	   #:defnode
	   #:defoptimizer
	   #:deftrainer
	   #:defdataset
	   #:WaffeDataset

	   #:save-for-backward
	   
	   #:step-model
	   #:predict
	   #:get-dataset
	   #:get-dataset-length

	   #:with-optimized-operation

	   #:model
	   #:update
	   #:zero-grad

	   #:model-list
	   #:with-model-list

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
