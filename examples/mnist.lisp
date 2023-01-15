
(defpackage :mnist-example
  (:use :cl :cl-waffe :cl-waffe.nn :cl-libsvm-format)
  (:export demo))

(in-package :mnist-example)

; this file is excluded from cl-waffe-test
; here's mnist example codes and benchmark

(defmodel MLP (activation)
  :parameters ((layer1   (denselayer (* 28 28) 512 T activation))
	       (dropout1 (dropout 0.5))
	       (layer2   (denselayer 512 256 T activation))
	       (layer3   (linearlayer 256 10 T)))
  :forward ((x)
	    (with-calling-layers x
	      (layer1 x)
	      (dropout1 x)
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

(defdataset Mnistdata (train valid batch-size)
  :parameters ((train train) (valid valid) (batch-size batch-size))
  :next    ((index)
	    (list (!set-batch (self train) index (self batch-size))
		  (!set-batch (self valid) index (self batch-size))))
  :length (() (car (!shape (self train)))))


(defmacro do-index-value-list ((index value list) &body body)
  (let ((iter (gensym))
        (inner-list (gensym)))
    `(labels ((,iter (,inner-list)
                     (when ,inner-list
                       (let ((,index (car ,inner-list))
                             (,value (cadr ,inner-list)))
                         ,@body)
                       (,iter (cddr ,inner-list)))))
       (,iter ,list))))

(defun read-data (data-path data-dimension n-class &key (most-min-class 1))
  (let* ((data-list (svmformat:parse-file data-path))
         (len (length data-list))
         (target     (make-array (list len n-class)
				       :element-type 'float
				       :initial-element 0.0))
         (datamatrix (make-array (list len data-dimension)
				       :element-type 'float
				       :initial-element 0.0)))
    (loop for i fixnum from 0
          for datum in data-list
          do (setf (aref target i (- (car datum) most-min-class)) 1.0)
             (do-index-value-list (j v (cdr datum))
               (setf (aref datamatrix i (- j most-min-class)) v)))
    (values (const (mgl-mat:array-to-mat datamatrix))
		   (const (mgl-mat:array-to-mat target)))))

(defun demo ()
  (multiple-value-bind (datamat target)
      (read-data "examples/tmp/mnist.scale" 784 10 :most-min-class 0)
    (defparameter mnist-dataset datamat)
    (defparameter mnist-target target))

  (multiple-value-bind (datamat target)
      (read-data "examples/tmp/mnist.scale.t" 784 10 :most-min-class 0)
    (defparameter mnist-dataset-test datamat)
    (defparameter mnist-target-test target))

  (format t "Training: ~a" (!shape mnist-dataset))
  (format t "Valid   : ~a" (!shape mnist-target))
  (print "")

  (setq trainer (MLPTrainer :relu 1e-4))

  (setq train (MnistData mnist-dataset mnist-target 100))
  (setq test (MnistData mnist-dataset-test mnist-target-test 100))

  (time (train trainer train :max-iterate 600 :epoch 10 :batch-size 100 :valid-dataset test :verbose t :random t)))

