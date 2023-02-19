
(in-package :cl-waffe-test)

(in-suite :test)


(defparameter x (parameter (!randn `(10 100))))

(defparameter linearlayer1 (linearlayer 100 10 t))
(defparameter linearlayer2 (linearlayer 100 10 nil))

(defparameter denselayer1 (denselayer 100 10 t :relu))
(defparameter denselayer2 (denselayer 100 10 t :sigmoid))
(defparameter denselayer3 (denselayer 100 10 t :tanh))
(defparameter denselayer4 (denselayer 100 10 t #'!tanh))

(defparameter dropout (dropout 0.5))
(defparameter batchnorm2d (dropout 0.5))


(defparameter embedding (Embedding 10 10))

(defparameter rnn1 (RNN 10 256 :num-layers 1))
(defparameter rnn2 (RNN 10 256 :num-layers 3))

(defparameter words (call embedding (parameter (!ones `(10 10)))))

(defparameter model-list (model-list (list (linearlayer 10 1)
					   (linearlayer 10 1))))

(defparameter model-list1 (mlist  (linearlayer 10 1)
				  (linearlayer 10 1)))

(defun test-model-list ()
  (call model-list (const 0) (!randn `(10 10)))
  (call (mth 1 model-list1) (!randn `(10 10)))
  t)

(defun test-model (model input)
  (let* ((i (parameter input))
	 (out (call model i)))
    (backward (!sum out))
    (grad i)))

(test networks-test
      (is (test-model linearlayer1 x))
      (is (test-model linearlayer2 x))
      (is (test-model denselayer1 x))
      (is (test-model denselayer2 x))
      (is (test-model denselayer3 x))
      (is (test-model denselayer4 x))
      (is (test-model dropout x))
      (is (test-model batchnorm2d x))
      ;(is (test-model embedding (parameter (!ones `(10 10)))))
      ;(is (test-model rnn1 words))
      ;(is (test-model rnn2 words))
      (is (test-model-list))
      )

