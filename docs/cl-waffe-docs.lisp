
(in-package :cl-waffe.documents)

(with-page *cl-waffe-doc* "Package :cl-waffe"
  (macrolet ((redirect (name &optional (comment ""))
	       `(insert (format nil "@link[uri=\"./<apis--cl-waffe>.html#~a\"](~a) ~a~%~%" ,name ,name ,comment)))
	     (with-section1 (title comment &body body)
	       `(with-section ,title
		  (insert "~%@begin(deflist)~%@def(~a)~%@begin(term)~%" ,comment)
		  ,@body
		  (insert "~%@end(term)~%@end(deflist)~%"))))

    (with-section1 "Sampling Distributions" "Initializes the tensor of the given dim with specified algorithms."
      (redirect "!normal")
      (redirect "!randn")
      (redirect "!uniform-random")
      (redirect "!beta")
      (redirect "!gamma")
      (redirect "!chisquare")
      (redirect "!bernoulli")
      (redirect "!binomial"))

    (with-section1 "Tensor Initializer" ""
      (redirect "!zeros")
      (redirect "!ones")
      (redirect "!fill")
      (redirect "!zeros-like")
      (redirect "!ones-like")
      (redirect "!full-like"))
      
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
  


  
  )
