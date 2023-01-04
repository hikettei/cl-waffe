
(in-package :cl-waffe-test)

(in-suite :test)

; simple mlp and test activations


(defmodel MLP (activation)
  :parameters ((layer1 (cl-waffe.nn:denselayer (* 28 28) 128 T activation))
	       (layer2 (cl-waffe.nn:denselayer 128 256 T activation))
	       (layer3 (cl-waffe.nn:denselayer 256 10 T :softmax)))
  :forward ((x)
	    (call (self layer3)
		  (call (self layer2)
			(call (self layer1) x)))))

(defun test-mlp (activation)
  (let* ((model (MLP activation))
	 (input (randn (* 28 28) 128))
	 (out (sum (call model input) 0)))
    (backward out)
    out))

(setq model (MLP :sigmoid))
(print-model model)

(setq input (randn (* 28 28) 2))
(setq out (sum (call model input) 0))

(setq optim (cl-waffe.optimizers:init-optimizer cl-waffe.optimizers:sgd model))

(backward out)
(call optim)

;(print n)
(print out)

;(test-mlp :sigmoid)
;(test-mlp :tanh)
;(test-mlp :relu)


;(test cl-waffe-test
;      (is (test-mlp :sigmoid))
;      (is (test-mlp :tanh))
;      (is (test-mlp :relu)))
