
(defpackage :rnn-example
  (:use :cl :cl-waffe :cl-waffe.nn)
  (:export demo))

(in-package :rnn-example)

(deftrainer RNNTrainer (input-size hidden-size)
  :model (RNN input-size hidden-size :num-layers 3)
  :optimizer cl-waffe.optimizers:Adam
  :optimizer (:lr 1e-4)
  :step-model ((x y)
	       (let ((out (softmax-cross-entropy
			   (call (model) x)
			   y)))
		 (backward out)
		 out))
  :predict ((x) (call (model) x)))

(defun demo ()
  (setq trainer (RNNTrainer 128 256))
  (setq x (!randn `(50 10 128)))
  (setq y (!ones `(50 10 128)))
  (time (print (step-model trainer x y))))
