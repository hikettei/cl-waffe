
(in-package :cl-waffe-test)

(in-suite :test)

(defmodel MLP (activation)
  :parameters ((layer1 (cl-waffe.nn:denselayer (* 28 28) 128 T activation))
	       (layer2 (cl-waffe.nn:denselayer 128 256 T activation))
	       (layer3 (cl-waffe.nn:denselayer 256 10 T activation)))
  :forward ((x)
	    (call (self layer3)
		  (call (self layer2)
			(call (self layer1) x)))))


(deftrainer MLPTrainer (activation lr)
  :model          (MLP activation)
  :optimizer      cl-waffe.optimizers:SGD
  :optimizer-args (:lr lr)
  :step-model ((x) (let ((out (call (model) x)))
		     (backward out)
		     (update)
		     (zero-grad)
		     x)))


(setq trainer (MLPTrainer :relu 1e-3))
(setq input (randn (* 28 28) 128))
(step-model trainer input)
