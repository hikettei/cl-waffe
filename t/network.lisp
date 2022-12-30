
(in-package :cl-waffe-test)

(in-suite :test)

; simple mlp

(defmodel MLP (activation)
  :parameters ((layer1 (cl-waffe.nn:denselayer (* 28 28) 128 activation))
	       (layer2 (cl-waffe.nn:denselayer 128 256 activation))
	       (layer3 (cl-waffe.nn:denselayer 256 10 activation)))
  :forward ((x)
	    (call (cl-waffe:self layer3)
		  (call (cl-waffe:self layer2)
			(call (cl-waffe:self layer1) x)))))

(setq model (MLP :sigmoid))
(setq input (randn (* 28 28) 128))

(setq out (sum (call model input) 0))

(print "Loss")
(print (data out))
(backward out)

