
(in-package :cl-waffe-test)

(in-suite :test)


(defparameter x (!randn `(10 100)))

(defparameter linearlayer1 (linearlayer 100 10 t))
(defparameter linearlayer2 (linearlayer 100 10 nil))

(defparameter denselayer1 (denselayer 100 10 t :relu))
(defparameter denselayer2 (denselayer 100 10 t :sigmoid))
(defparameter denselayer3 (denselayer 100 10 t :tanh))
(defparameter denselayer4 (denselayer 100 10 t #'!tanh))

(defparameter dropout (dropout 0.5))

(defparameter model-list (model-list (list (linearlayer 10 1)
					   (linearlayer 10 1))))

(defparameter model-list1 (mlist  (linearlayer 10 1)
				  (linearlayer 10 1)))

(defun test-model-list ()
  (call model-list (const 0) (!randn `(10 10)))
  (call (mth 1 model-list1) (!randn `(10 10)))
  t)

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
      (is (test-model denselayer4 x))
      (is (test-model dropout x))
      (is (test-model-list))
      )

