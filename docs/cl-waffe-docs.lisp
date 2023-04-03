
(in-package :cl-waffe.documents)

(with-page *cl-waffe-doc* "Package :cl-waffe"
  )

(with-page *APIS-cl-waffe* "<APIs: cl-waffe>"

  ; distributions
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

  (with-api "function" "!shape")
  (with-api "function" "!dims")
  (with-api "function" "!size")
  (with-api "function" "!zeros-like")
  (with-api "function" "!ones-like")
  (with-api "function" "!full-like")
  


  
  )
