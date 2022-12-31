
(in-package :cl-waffe-test)

(in-suite :test)

(defmodel MLP (activation)
  :parameters ((layer1 (cl-waffe.nn:denselayer (* 28 28) 128 T activation))
	       (layer2 (cl-waffe.nn:denselayer 128 256 T activation))
	       (layer3 (cl-waffe.nn:linearlayer 256 10 T)))
  :forward ((x)
	    (call (self layer3)
		  (call (self layer2)
			(call (self layer1) x)))))

(deftrainer MLPTrainer (activation lr)
  :model          (MLP activation)
  :optimizer      cl-waffe.optimizers:SGD
  :optimizer-args (:lr lr)
  :step-model ((x y)
	       (let ((out (cl-waffe.nn:mse (call (model) x) y)))
		 (backward out)
		 (update)
		 (zero-grad)
		 out)))

(defdataset Mnistdata (train valid)
  :parameters ((train train) (valid valid))
  :forward ((index) (list (array-ref (self train) index t)
			  (array-ref (self valid) index t)))
  :length (() (car (shape (self train)))))

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
         (target     (numcl:zeros (list len n-class)))
         (datamatrix (numcl:zeros (list len data-dimension))))
    (loop for i fixnum from 0
          for datum in data-list
          do (setf (numcl:aref target i (- (car datum) most-min-class)) 1.0)
             (do-index-value-list (j v (cdr datum))
               (setf (numcl:aref datamatrix i (- j most-min-class)) v)))
    (values (const datamatrix) (const target))))

(multiple-value-bind (datamat target)
    (read-data "t/tmp/mnist.scale" 784 10 :most-min-class 0)
  (defparameter mnist-dataset datamat)
  (defparameter mnist-target target))

(multiple-value-bind (datamat target)
    (read-data "t/tmp/mnist.scale.t" 784 10 :most-min-class 0)
  (defparameter mnist-dataset-test datamat)
  (defparameter mnist-target-test target))

(format t "Training: ~a" (shape mnist-dataset))
(format t "Valid   : ~a" (shape mnist-target))

(setq trainer (MLPTrainer :sigmoid 1e-3))

(setq train (MnistData mnist-dataset mnist-target))
(setq valid (MnistData mnist-dataset-test mnist-target-test))

(train trainer train :max-iterate 20)


