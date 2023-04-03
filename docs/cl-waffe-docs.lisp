
(in-package :cl-waffe.documents)

(with-page *cl-waffe-doc* "Package :cl-waffe"
  (macrolet ((redirect (name &optional (comment ""))
	       `(insert (format nil "@link[uri=\"./<apis--cl-waffe>.html#~a\"](~a) ~a~%~%" ,name ,name ,comment)))
	     (with-section1 (title comment &body body)
	       `(with-section ,title
		  (insert "~%@begin(deflist)~%@def(~a)~%@begin(term)~%" ,comment)
		  ,@body
		  (insert "~%@end(term)~%@end(deflist)~%"))))

    (with-section1 "Tensors" ""
      (redirect "WaffeTensor")
      (redirect "const")
      (redirect "tensor")
      (redirect "sysconst")
      (redirect "parameter")
      (redirect "data")
      (redirect "value")
      (redirect "grad")
      (redirect "backward"))

    (with-section1 "Gradients" ""
      (redirect "with-no-grad")
      (redirect "*no-grad*"))
    
    (with-section1 "Four Arithmetic Operations" "Element wise operations"
      (redirect "!add" "Adds the given tensors element by element.")
      (redirect "!sub" "Substracts the given tensors element by element.")
      (redirect "!mul" "Multiply the given tensors element by element.")
      (redirect "!div" "Divides the given tensors element by element."))

    (with-section1 "Summarize" "Get a scalar value from a matrix"
      (redirect "!sum")
      (redirect "!mean"))

    (with-section1 "Multiplying matrices" ""
      (redirect "!dot")
      (redirect "!matmul")
      (redirect "!einsum" "⚠️ It is unavailable."))

    (with-section1 "Trigonometric Functions" ""
      (redirect "!sin")
      (redirect "!cos")
      (redirect "!tan")
      (redirect "!asin")
      (redirect "!acos")
      (redirect "!atan")
      (redirect "!sinh")
      (redirect "!cosh")
      (redirect "!tanh")
      (redirect "!asinh")
      (redirect "!acosh")
      (redirect "!atanh"))

    (with-section1 "Mathematical Functions" ""
      (redirect "!abs")
      (redirect "!log")
      (redirect "!exp")
      (redirect "!pow")
      (redirect "!sqrt")
      (redirect "!argmax")
      (redirect "!argmin"))

    (with-section1 "Reshaping" ""
      (redirect "!squeeze")
      (redirect "!unsqueeze")
      (redirect "!reshape")
      (redirect "!repeats")
      (redirect "!flatten")
      (redirect "!transpose" "Often used with !matmul")
      (redirect "!transpose1" "Forced to transpose."))

    (with-section1 "Shaping" "Accessing the shape"
      (redirect "!shape")
      (redirect "!dims")
      (redirect "!size"))

    (with-section1 "Concatenate and Split" ""
      (redirect "!concatenate")
      (redirect "!stack")
      (redirect "!split")
      (redirect "!vstack")
      (redirect "!hstack"))

    (with-section1 "Iterations and Making Copy" ""
      (redirect "!aref" "it behaves as if aref, but works like numpy")
      (redirect "!displace" "TODO")
      (redirect "!set-batch" "displace the tensor")
      (redirect "!where")
      (redirect "!index")
      (redirect "!filter"))
      
    
    (with-section1 "Sampling Distributions" "Initializes the tensor of the given dim with specified algorithms."
      (redirect "!normal" "Sampling the standard distribution")
      (redirect "!randn")
      (redirect "!uniform-random" "Sampling the uniform random")
      (redirect "!beta" "Sampling the beta distribution")
      (redirect "!gamma" "Sampling the gamma distribution")
      (redirect "!chisquare" "Sampling the chisquare distribution")
      (redirect "!bernoulli" "Sampling the bernoulli distribution")
      (redirect "!binomial"))

    (with-section1 "Initializes the tensor" "Tensors with the same elements."
      (redirect "!zeros")
      (redirect "!ones")
      (redirect "!fill" "Fill up a tensor with specified value")
      (redirect "!zeros-like" "Returns a tensor with the same dimension as the given dimension but elements are zero.")
      (redirect "!ones-like")
      (redirect "!full-like")
      (redirect "!arange"))

    (with-section1 "Activations" ""
      (redirect "!tanh")
      (redirect "!sigmoid")
      (redirect "!relu")
      (redirect "!gelu" "Not Tested")
      (redirect "!leakey-relu" "Not Tested")
      (redirect "!swish" "Not Tested")
      (redirect "!mish" "TODO")
      (redirect "!softmax"))

    (with-section1 "Logging" ""
      (redirect "with-verbose" "Displays Computation Node"))

    (with-section1 "Dtype" ""
      (redirect "with-dtype")
      (redirect "dtypecase")
      (redirect "define-with-typevar"))

    (with-section1 "Extensions" ""
      (redirect "with-backend")
      (redirect "define-node-extension")
      (redirect "*restart-non-exist-backend*"))

    (with-section1 "Destructive Operations" ""
      (redirect "!allow-destruct")
      (redirect "!disallow-destruct"))

    (with-section1 "Objects" "defmodel/defnode/defoptimizer"
      (redirect "defnode")
      (redirect "defmodel")
      (redirect "defoptimizer")
      (redirect "call")
      (redirect "call-backward")
      (redirect "self")
      (redirect "save-for-backward")
      (redirect "get-forward-caller")
      (redirect "get-backward-caller")
      (redirect "with-calling-layers")
      )

    (with-section1 "Trainer" ""
      (redirect "deftrainer")
      (redirect "step-model")
      (redirect "predict")
      (redirect "update")
      (redirect "zero-grad")
      )

    (with-section1 "Dataset" ""
      (redirect "defdataset")
      (redirect "get-dataset")
      (redirect "get-dataset-length"))

    (with-section1 "Model List" ""
      (redirect "mlist")
      (redirect "model-list")
      (redirect "mlist"))

    (with-section1 "Printing" ""
      (redirect "print-model")
      (redirect "*print-char-max-size*")
      (redirect "*print-arr-max-size*")
      (redirect "*print-mat-max-size*")
      (redirect "*ignore-inlining-info*"))
    
    ))

(with-page *APIS-cl-waffe* "<APIs: cl-waffe>"

  ; Initializers/distributions of tensor.
  (with-api "function" "!normal"
    (with-evals "(!normal `(10 10)) :mean 2.0 :stddev 1.0"))

  (with-api "function" "!randn"
    (with-evals "(!randn `(10 10))"))

  (with-api "function" "!uniform-random"
    (with-evals "(!uniform-random `(10 10) :limit 2.0)"))

  (with-api "function" "!beta"
    (with-evals "(!beta `(10 10) 5.0 1.0)"))

  (with-api "function" "!gamma"
    (with-evals "(!gamma `(10 10) 1.0)"))

  (with-api "function" "!chisquare"
    (with-evals "(!chisquare `(10 10) 2.0)"))

  (with-api "function" "!bernoulli"
    (with-evals "(!bernoulli `(10 10) 0.5)"))

  (with-api "function" "!binomial"
    (with-evals "(!binomial `(10 10) 0.5)"))

  (with-api "function" "!random-with"
    (with-evals "(!random-with '(10 10) #'(lambda (n) n))"))

  (with-api "function" "!random"
    (with-evals
      "(!random `(10 10) 1.0)"
      "(!random `(10 10) 3)"
      "(!random `(10 10) `(1.0 2.0))"
      "(!random `(10 10) `(1 5))"))

  (with-api "function" "!zeros-like")
  (with-api "function" "!ones-like")
  (with-api "function" "!full-like")

  (with-api "function" "!zeros")
  (with-api "function" "!ones")
  (with-api "function" "!fill")

  ; accessors
  (with-api "function" "!shape")
  (with-api "function" "!dims")
  (with-api "function" "!size")

  (with-api "struct" "WaffeTensor")
  (with-api "macro" "parameter")
  (with-api "function" "data")
  (with-api "function" "value")
  (with-api "function" "backward")

  (with-api "macro" "with-no-grad")
  (with-api "variable" "*no-grad*")

  (with-api "function" "!add")
  (with-api "function" "!!add")
  (with-api "function" "!sub")
  (with-api "function" "!!sub")
  (with-api "function" "!mul")
  (with-api "function" "!!mul")
  (with-api "function" "!div")

  (with-api "function" "!sum")
  (with-api "function" "!mean")

  (with-api "function" "!dot")
  (with-api "function" "!matmul")

  (with-api "function" "!sin")
  (with-api "function" "!cos")
  (with-api "function" "!tan")

  (with-api "function" "!asin")
  (with-api "function" "!acos")
  (with-api "function" "!atan")

  (with-api "function" "!sinh")
  (with-api "function" "!cosh")
  (with-api "function" "!tanh")

  (with-api "function" "!asinh")
  (with-api "function" "!acosh")
  (with-api "function" "!atanh")

  
  (with-api "function" "!abs")
  (with-api "function" "!log")
  (with-api "function" "!exp")

  (with-api "function" "!pow")
  (with-api "function" "!sqrt")

  (with-api "function" "!argmax")
  (with-api "function" "!argmin")

  (with-api "function" "!squeeze")
  (with-api "function" "!unsqueeze")

  (with-api "function" "!reshape")
  (with-api "function" "!repeats")
  (with-api "function" "!flatten")
  (with-api "function" "!transpose")
  (with-api "function" "!transpose1")

  (with-api "function" "!concatenate")
  (with-api "function" "!stack")

  (with-api "function" "!split")

  (with-api "macro" "!hstack")
  (with-api "macro" "!vstack")

  (with-api "function" "!aref")
  ;(with-api "function" "!displace")
  (with-api "function" "!where")
  (with-api "function" "!index")
  (with-api "function" "!filter")
  
  (with-api "macro" "!arange")

  (with-api "function" "!relu")
  (with-api "function" "!sigmoid")
  (with-api "function" "!gelu")
  (with-api "function" "!leakey-relu")
  (with-api "function" "!swish")
  (with-api "function" "!softmax")

  (with-api "macro" "with-verbose")
  (with-api "macro" "with-dtype")
  (with-api "macro" "dtypecase")
  (with-api "macro" "define-with-typevar")

  (with-api "macro" "with-backend")
  (with-api "macro" "define-node-extension")
  (with-api "variable" "*restart-non-exist-backend*")

  (with-api "macro" "!allow-destruct")
  (with-api "macro" "!disallow-destruct")

  (with-api "macro" "defnode")
  (with-api "macro" "defmodel")
  (with-api "macro" "defoptimizer")

  (with-api "macro" "call")
  (with-api "macro" "call-backward")

  (with-api "macro" "self")
  (with-api "macro" "save-for-backward")
  (with-api "macro" "get-forward-caller")
  (with-api "macro" "get-backward-caller")
  (with-api "macro" "with-calling-layers")

  (with-api "macro" "deftrainer")
  (with-api "function" "step-model")
  (with-api "function" "predict")

  (with-api "macro" "model")
  (with-api "macro" "update")
  (with-api "macro" "zero-grad")

  (with-api "macro" "defdataset")

  (with-api "function" "get-dataset")
  (with-api "function" "get-dataset-length")

  (with-api "function" "model-list")
  (with-api "function" "mlist")
  (with-api "function" "mth")

  (with-api "macro" "grad")
  )
