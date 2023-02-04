
(defpackage :mnist-example
  (:use :cl :cl-waffe :cl-waffe.nn :cl-waffe.io)
  (:export demo))

(in-package :mnist-example)

; this file is excluded from cl-waffe-test
; here's mnist example codes and benchmark

(defmodel MLP (activation)
  :parameters ((layer1   (denselayer (* 28 28) 512 T activation))
	       (layer2   (denselayer 512 256 T activation))
	       (layer3   (linearlayer 256 10 T)))
  :forward ((x)
	    (with-calling-layers x
	      (layer1 x)
 	      (layer2 x)
	      (layer3 x))))

(deftrainer MLPTrainer (activation lr)
  :model          (MLP activation)
  :optimizer      cl-waffe.optimizers:Adam
  :optimizer-args (:lr lr)
  :step-model ((x y)
	       (zero-grad)
	       (let ((out (cl-waffe.nn:softmax-cross-entropy (call (model) x) y)))
		 (backward out)
		 (update)
		 out))
 :predict ((x) (call (model) x)))

(defun demo () (time (demo1)))
(defun demo1 ()
  (defparameter batch-size 300)
  
  (sb-profile:profile mgl-mat::blas-sgemm
		      mgl-mat::blas-scopy
		      mgl-mat::array-to-mat
		      cl-waffe::backward1
		      cl-waffe.nn::softmax-cross-entropy
		      cl-waffe::!sum
		      cl-waffe::!mul
		      cl-waffe::!div
		      cl-waffe::!add
		      cl-waffe::!matmul
		      cl-waffe::!relu
		      cl-waffe::!exp
		      cl-waffe::!softmax
		      cl-waffe::!faref
		      cl-waffe::!write-faref
		      cl-waffe.backends.mgl::adam-update
		      svmformat:parse-file)

  (setq trainer (MLPTrainer :relu 1e-4))

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

  #|
  (defparameter mnist-dataset (!ones `(200 784)))
  (defparameter mnist-target  (!randn `(200 10)))

  (defparameter mnist-dataset-test (!zeros `(100 784)))
  (defparameter mnist-target-test (!zeros `(100 10)))
  |#

  (format t "Training: ~a" (!shape mnist-dataset))
  (format t "Valid   : ~a" (!shape mnist-target))
  (print "")

  (setq train (WaffeDataSet mnist-dataset
			    mnist-target
			   :batch-size batch-size))
  (setq test (WaffeDataSet mnist-dataset-test
			   mnist-target-test
			   :batch-size batch-size))
  
  (mgl-mat:with-mat-counters (:count count :n-bytes n-bytes)
    (time (train trainer train :max-iterate 600 :epoch 100 :batch-size batch-size :valid-dataset test
			       :verbose t :random t :print-each 100))
    (format t "Count: ~a~%" count)
    (format t "Consumed: ~abytes~%" n-bytes))

  (sb-profile:report))

