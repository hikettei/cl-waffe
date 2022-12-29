
(in-package :cl-waffe-test)

(in-suite :test)

; simple mlp

(defmodel MLP nil
  :parameters ((layer1 (cl-waffe.nn:denselayer (* 28 28) 128))
	       (layer2 (cl-waffe.nn:denselayer 128 256))
	       (layer3 (cl-waffe.nn:denselayer 256 10)))
  :forward ((x)
	    (call (cl-waffe:self layer3)
		  (call (cl-waffe:self layer2)
			(call (cl-waffe:self layer1) x)))))

(setq model (MLP))
(setq input (randn (* 28 28) 128))
(setq out (mean (call model input) 0))
(print (data out))
(backward out)
