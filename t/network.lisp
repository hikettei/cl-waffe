
(in-package :cl-waffe-test)

(in-suite :test)

(defparameter x (!randn `(10 100)))

(defparameter linearlayer1 (linearlayer 100 10 t))
(defparameter linearlayer2 (linearlayer 100 10 nil))

(defparameter denselayer1 (denselayer 100 10 t :relu))
(defparameter denselayer2 (denselayer 100 10 t :sigmoid))
(defparameter denselayer3 (denselayer 100 10 t :tanh))
(defparameter denselayer4 (denselayer 100 10 t #'!tanh))

(defun test-model (model x)
  (let ((out (call model x)))
    (backward (!sum out))
    t))

(test networks-test
      (is (test-model linearlayer1 x))
      (is (test-model linearlayer2 x))
      (is (test-model denselayer1 x))
      (is (test-model denselayer2 x))
      (is (test-model denselayer3 x))
      (is (test-model denselayer4 x)))
 
