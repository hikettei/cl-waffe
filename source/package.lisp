
(in-package :cl-user)

(defpackage cl-waffe
  (:documentation "An package for defining node, initializing and computing with tensor, and backprops.")
  (:use :cl :mgl-mat :alexandria :inlined-generic-function)
  (:export #:*verbose*
	   #:with-verbose)
  (:export #:with-jit
	   #:with-jit-debug)
  (:export #:*single-value*)
  (:export #:waffetensor
	   #:tensor
	   #:const
	   #:sysconst
	   #:parameters
	   #:hide-from-tree
	   #:forward
	   #:backward)
  (:export 
	   ; An parameters for displaying tensor.
	   #:*print-char-max-len*
	   #:*print-arr-max-size*
	   #:*print-mat-max-size*)

  (:export
	   #:!allow-destruct
	   #:!disallow-destruct)

  (:export
	   ; Accessors
           #:warranty
           #:waffe-tensor-p	   
	   #:data
	   #:value
	   #:grad)
  (:export
	   #:waffedatatype
	   #:waffe-array)
  (:export
   #:waffetensor-thread-data
   #:waffetensor-is-next-destruct?)

  (:export
	   #:with-no-grad
	   #:*no-grad*)

  (:export
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
	   #:with-optimized-operation)

  (:export ; macros in waffe-object
           #:model
	   #:self
	   #:save-for-backward
	   #:update
	   #:zero-grad)

  (:export
           #:mlist
	   #:mth
	   #:model-list
	   #:with-model-list)

  (:export
	   #:train
	   #:get-dataset
	   #:get-dataset-length)

  (:export 
	   #:parameter
	   #:call
	   #:backward)

  (:export
	   #:!set-batch
	   #:!reset-batch)

  (:export
	   #:waffetensor-destructively-calln
	   #:waffetensor-destructive?
	   #:waffetensor-is-data-destructed?
	   #:waffetensor-report-index
	   #:with-ignore-optimizer
	   #:*ignore-optimizer*)

  (:export
	   #:print-model
	   #:*default-backend*
	   #:extend-from)

  (:export
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
	   #:with-calling-layers)

  (:export
	   #:!add
	   #:!!add
	   #:!sub
	   #:!!sub
	   #:!mul
	   #:!!mul
	   #:!div
	   #:!!div)

  (:export
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
	   #:!abs)

  (:export
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
	   #:!argmax)

  (:export
	   
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
	   #:!gelu)

  (:export
	   #:!concatenate
	   #:!split
	   #:!stack
	   #:!vstack
	   #:!hstack

	   #:with-usage))

(defparameter *cl-waffe-object-types* `(:model
					:node
					:trainer
					:optimizer
					:dataset)
  "An identifiers of cl-waffe's objects.")

(defmacro save-for-backward (slot tensor)
  (error "Welcome to cl-waffe.
Attempting your tensor ~a to a slot ~a, but save-for-backward wasn't called in:
1. defmodel's forward or backward.
2. defnode's forward or backward.

save-for-backward is useful when registering temporary tensor depending on the case when copied tensor will be used when backwards.

For example:

(defnode XXX nil
:forward ((x)
          (!exp x) ; <- (!exp x) doesn't copies x
..."
	 slot
	 tensor))

(defmacro self (name)
  (error "Welcome to cl-waffe.
Attempting to access ~a but couldn't.
This is because self wasn't called in:
1. defmodel's forward or backward
2. defnode's forward or backward
3. defoptimizer's update
4. defdataset's slots
5. deftrainer's slots

By using self, you can access cl-waffe's model parameter.
For example:

(defmodel XXX nil
  :parameters ((A 0))
  :forward ((x)
            (+ (self A) x)))" name))

(defmacro model ()
  (error "Welcome to cl-waffe.
Attempting to access the currently model but (model) wasn't called in:
1. defmodel/defnode's forward or backward
2. defoptimizer's slots"))

(defmacro update (&rest args)
  (declare (ignore args))
  (error "Welcome to cl-waffe.
The macro (update) can only be called in deftrainer's slots.
For details, documentations are available.

https://hikettei.github.io/cl-waffe-docs/docs/cl-waffe.html#3-deftrainer"))

(defmacro zero-grad ()
  (error "Welcome to cl-waffe.
The macro (zero-grad) can only be called in deftrainer's slots.
For details, documentations are available.

https://hikettei.github.io/cl-waffe-docs/docs/cl-waffe.html#3-deftrainer"))
