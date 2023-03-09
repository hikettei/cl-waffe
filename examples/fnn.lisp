
(defpackage :fnn-example
  (:use :cl :cl-waffe :cl-waffe.nn :cl-waffe.io)
  (:export demo))

(in-package :fnn-example)

(defmodel fnn (in-features hidden-size &key (dropout-rate 0.2))
  :parameters ((layer (linearlayer in-features hidden-size t))
	       (dropout (dropout dropout-rate)))
  :forward ((x)
	    (!relu
	     (with-calling-layers x
	       (layer x)
	       (dropout x)))))

(defmodel fnn-models (in-features)
  :parameters ((layer1 (FNN in-features 256 :dropout-rate 0.2))
	       (layer2 (FNN 256 256 :dropout-rate 0.2))
	       (layer3 (linearlayer 256 10 t))
	       (dropout (dropout 0.2)))
  :forward ((x)
	    (with-calling-layers x
	      (layer1 x)
	      (layer2 x)
	      (layer3 x)
	      (dropout x))))

(deftrainer fnn-mnist-trainer (in-features lr)
  :model (FNN-Models in-features)
  :optimizer cl-waffe.optimizers:Adam
  :optimizer-args (:lr lr)
  :step-model ((x y)
	       (zero-grad)
	       (let ((out (softmax-cross-entropy
			   (call (model) x)
			   y)))
		 (backward out)
		 (update)
		 out))
  :predict ((x)
	    (call (model) x)))

(defun demo (&key (batch-size 100))
  (let ((train)
	(test)
	(trainer))
    (format t "Loading examples/tmp/mnist.scale ...~%")
    
    (multiple-value-bind (datamat target)
	(read-libsvm-data "examples/tmp/mnist.scale" 784 10 :most-min-class 0)
      (defparameter mnist-dataset datamat)
      (defparameter mnist-target target))

    (format t "Loading examples/tmp/mnist.scale.t~%")

    (multiple-value-bind (datamat target)
	(read-libsvm-data "examples/tmp/mnist.scale.t" 784 10 :most-min-class 0)
      (defparameter mnist-dataset-test datamat)
      (defparameter mnist-target-test target))

    (format t "Training: ~a~%" (!shape mnist-dataset))
    (format t "Valid   : ~a~%" (!shape mnist-target))
    (format t "Test    : ~a~%"  (!shape mnist-dataset-test))

    (setq train (WaffeDataSet mnist-dataset
			      mnist-target
			      :batch-size batch-size))
    
    (setq test (WaffeDataSet mnist-dataset-test
			     mnist-target-test
			     :batch-size 100))

    ; I wish there were ReduceLROnPlateau ... (When using adam)
    (setq trainer (fnn-mnist-trainer (* 28 28) 1e-3))
    (time
     (train
      trainer
      train
      :max-iterate 600
      :epoch 30
      :batch-size batch-size
      :valid-dataset test
      :verbose t :random t :print-each 100))))
