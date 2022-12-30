
(in-package :cl-waffe-test)

(in-suite :test)

; simple mlp and test activations

(defmodel MLP (activation)
  :parameters ((layer1 (cl-waffe.nn:denselayer (* 28 28) 128 T activation))
	       (layer2 (cl-waffe.nn:denselayer 128 256 T activation))
	       (layer3 (cl-waffe.nn:denselayer 256 10 T activation)))
  :forward ((x)
	    (call (cl-waffe:self layer3)
		  (call (cl-waffe:self layer2)
			(call (cl-waffe:self layer1) x)))))

(defun test-mlp (activation)
  (let* ((model (MLP activation))
	 (input (randn (* 28 28) 128))
	 (out (sum (call model input) 0)))
    (backward out)
    out))

(test-mlp :sigmoid)
(test-mlp :tanh)
(test-mlp :relu)


;(test cl-waffe-test
;      (is (test-mlp :sigmoid))
;      (is (test-mlp :tanh))
;      (is (test-mlp :relu)))
