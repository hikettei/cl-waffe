
(in-package :cl-user)

(defpackage cl-waffe
  (:documentation "An package for defining node, initializing and computing with tensor, and backprops.")
  (:use :cl :mgl-mat :alexandria :inlined-generic-function)
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

	   #:!allow-destruct
	   #:!disallow-destruct

	   ; Functions for using tensor's data
	   #:warranty
	   #:data
	   #:value
	   #:grad

	   #:*in-node-method*

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
	   #:reset-config
	   
	   #:step-model
	   #:predict
	   #:get-dataset
	   #:get-dataset-length

	   #:call-and-dispatch-kernel
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
	   #:!init-with
	   #:!normal
	   #:!randn
	   #:!bernoulli
	   #:!binomial
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

	   #:!filter
	   #:with-calling-layers

	   #:!add
	   #:!!add
	   #:!sub
	   #:!!sub
	   #:!mul
	   #:!!mul
	   #:!div
	   #:!!div

	   #:!dot
	   #:!sum

	   #:!sqrt
	  
	   #:!pow
	   #:!mean
	   #:!log
	   #:!reshape
	   #:!transpose
	   #:!transpose1
	   #:!exp
	   #:!matmul
	   #:!repeats
	   #:!abs

	   #:!sin
	   #:!cos
	   #:!tan
	   #:!asin
	   #:!acos
	   #:!atan
	   #:!sinh
	   #:!cosh
	   #:!asinh
	   #:!acosh
	   #:!atanh

	   #:!argmin
	   #:!argmax
	   
	   #:!squeeze
	   #:!unsqueeze
	   #:!flatten
	   #:!modify
	   
	   #:!relu
	   #:!sigmoid
	   #:!tanh
	   #:!softmax
	   #:!leakey-relu
	   #:!swish
	   #:Swish
	   #:!gelu

	   #:with-usage
	   #:mlist
	   #:mth

	   ))

(defparameter *cl-waffe-object-types* `(:model
					:node
					:trainer
					:optimizer
					:dataset)
  "An identifiers of cl-waffe's objects.")
