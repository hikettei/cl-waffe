
(in-package :cl-waffe-test)

(in-suite :test)


(setq pallet (cl-termgraph:make-listplot-frame 40 14))
(setq l1 `(10 10 10 9.5 8 8 8 8 8 7 7 7 6 5 5 4 3 3 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 0.5))
(setq l2 `(11 11 11 9.5))

(cl-termgraph:init-line pallet :white)
(cl-termgraph:listplot-write pallet l1 :blue)
(cl-termgraph:listplot-write pallet l2 :red)

(cl-termgraph:listplot-print pallet :x-label "n" :y-label "loss"
				    :title "Test"
				    :descriptions `((:red "prev-loss" 0 4)
						    (:blue "current-loss" 0 3)))

(defmodel MLP (activation)
  :parameters ((layer1 (cl-waffe.nn:denselayer (* 28 28) 512 NIL activation))
	       (layer2 (cl-waffe.nn:denselayer 512 256 NIL activation))
	       (layer3 (cl-waffe.nn:denselayer 256 10 NIL activation)))
  :forward ((x)
	    (call (self layer3)
		  (call (self layer2)
			(call (self layer1) x)))))

(deftrainer MLPTrainer (activation lr)
  :model          (MLP activation)
  :optimizer      cl-waffe.optimizers:SGD
  :optimizer-args (:lr lr)
  :step-model ((x y)
	       (let ((out (cl-waffe.nn:cross-entropy (call (model) x) y)))
		 (backward out)
		 (update)
		 (zero-grad)
		 out)))

(defdataset Mnistdata (train valid)
  :parameters ((train train) (valid valid))
  :forward ((index) (list (!aref (self train) index t)
			  (!aref (self valid) index t)))
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
				       :element-type 'single-float
				       :initial-element 0.0))
         (datamatrix (make-array (list len data-dimension)
				       :element-type 'single-float
				       :initial-element 0.0)))
    (loop for i fixnum from 0
          for datum in data-list
          do (setf (aref target i (- (car datum) most-min-class)) 1.0)
             (do-index-value-list (j v (cdr datum))
               (setf (aref datamatrix i (- j most-min-class)) v)))
    (values (const (mgl-mat:array-to-mat datamatrix))
		   (const (mgl-mat:array-to-mat target)))))

(multiple-value-bind (datamat target)
    (read-data "t/tmp/mnist.scale" 784 10 :most-min-class 0)
  (defparameter mnist-dataset datamat)
  (defparameter mnist-target target))

(multiple-value-bind (datamat target)
    (read-data "t/tmp/mnist.scale.t" 784 10 :most-min-class 0)
  (defparameter mnist-dataset-test datamat)
  (defparameter mnist-target-test target))

(format t "Training: ~a" (!shape mnist-dataset))
(format t "Valid   : ~a" (!shape mnist-target))

(setq trainer (MLPTrainer :sigmoid 1e-3))

(setq train (MnistData mnist-dataset mnist-target))
(setq valid (MnistData mnist-dataset-test mnist-target-test))

(train trainer train :max-iterate 4 :epoch 60)


