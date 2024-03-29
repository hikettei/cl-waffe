
(in-package :cl-user)

(defpackage cl-waffe
  (:documentation "An package for defining node, initializing and computing with tensor, and backprops.")
  (:use :cl :mgl-mat :alexandria)

  (:export #:*verbose*
	   #:with-verbose)
  (:export #:with-jit
	   #:with-jit-debug)
  (:export #:*single-value*)
  (:export #:*lparallel-kernel*
	   #:*ignore-inlining-info*
	   #:set-lparallel-kernel)
  (:export #:with-dtype
	   #:dtypecase
	   #:*dtype*
	   #:define-with-typevar)
  (:export #:waffetensor
	   #:tensor
	   #:const
	   #:sysconst
	   #:parameters
	   #:hide-from-tree
	   #:forward
	   #:backward
	   #:lazy-transpose-p
	   #:maybe-copy)
  (:export #:with-backend
	   #:define-node-extension)
  (:export ; conditions
   #:invaild-slot-name-waffe-object
   #:shaping-error
   #:aref-shaping-error
   #:node-error
   #:Backend-Doesnt-Exists
   #:backward-error
   #:unimplemented-error
   #:nosuchnode-error)
  (:export 
	   ; An parameters for displaying tensor.
	   #:*print-char-max-len*
	   #:*print-arr-max-size*
	   #:*print-mat-max-size*)

  (:export
           #:WITH-SEARCHING-CALC-NODE
	   #:!allow-destruct
	   #:!disallow-destruct)

  (:export #:warranty
	   ; Accessors	   
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
	   #:*no-grad*
	   #:*restart-non-exist-backend*)
  (:export #:defmodel
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

	   #:call-and-dispatch-kernel)

  (:export ; macros (binded by defobject) in waffe-object
           #:model
	   #:self
	   #:save-for-backward
	   #:update
	   #:zero-grad)

  (:export
           #:mlist
	   #:mth
	   #:model-list)

  (:export
          #:get-forward-caller
          #:get-backward-caller)

  (:export
	   #:train
	   #:get-dataset
	   #:get-dataset-length)

  (:export 
	   #:parameter
	   #:call
	   #:call-backward
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
	   #:!uniform-random
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
	   #:!argmax

	   #:init-features)

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

(in-package :cl-waffe)

(defvar *call-forward-features* (common-lisp:make-hash-table)
  "An hash-table which records all forward nodes")
(defvar *call-backward-features* (common-lisp:make-hash-table)
  "An hash-table which records all backward nodes")

(defun init-features ()
  "Initializes all features."
  (format t "[WARN] Initializing features...")
  (setf *call-forward-features* (common-lisp:make-hash-table))
  (setf *call-forward-features* (common-lisp:make-hash-table))
  t)

(defparameter *cl-waffe-object-types* `(:model
					:node
					:trainer
					:optimizer
					:dataset)
  "An identifiers of cl-waffe's objects.")

(defmacro save-for-backward (slot tensor)
  "TODO :DOCSTRING"
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
  "Todo: Docstring"
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
  "Todo: Docstring"
  (error "Welcome to cl-waffe.
Attempting to access the currently model but (model) wasn't called in:
1. defmodel/defnode's forward or backward
2. defoptimizer's slots"))

(defmacro update (&rest args)
  "Todo: Docstring"
  (declare (ignore args))
  (error "Welcome to cl-waffe.
The macro (update) can only be called in deftrainer's slots.
For details, documentations are available.

https://hikettei.github.io/cl-waffe-docs/docs/cl-waffe.html#3-deftrainer"))

(defmacro zero-grad ()
  "Todo docstring"
  (error "Welcome to cl-waffe.
The macro (zero-grad) can only be called in deftrainer's slots.
For details, documentations are available.

https://hikettei.github.io/cl-waffe-docs/docs/cl-waffe.html#3-deftrainer"))
