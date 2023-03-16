
(in-package :cl-waffe.nn)

; Todo Implement Backward
(defmodel BatchNorm2d (in-features &key (affine t) (epsilon 1e-7))
  :document (with-usage "BatchNorm2d"
	      :note "todo: docs")
  :optimize t
  :parameters ((affine (if affine
			   (linearlayer in-features in-features T)
			   T))
	       (epsilon epsilon :type float))
  :forward ((x)
	    (let* ((average (!mean x 1 nil)) ; minibatch-average
		   (dist (!mean (!pow (!sub x average) 2.0) 1)) ; minibatch-dist
		   (r (!div (!sub x average)
			    (!sqrt (!add dist (self epsilon))))))
	      (if (eql (self affine) T)
		  r
		  (call (self affine) r)))))

; Todo LayerNorm (when implementing lstm/rnn)
