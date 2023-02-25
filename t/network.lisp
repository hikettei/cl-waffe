
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

(defparameter rnn1 (RNN 10 25 :num-layers 1))
(defparameter rnn2 (RNN 10 25 :num-layers 3))

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
  (format t "~%Running test forward and backward of ~a~%" model)
  (format t "~%Calling Forward:~%")
  (let* ((i (parameter input))
	 (out (time (call model i))))
    (format t "~%Calling Backward:~%")
    (time (backward (!sum out)))
    (grad i)))

(defmodel Encoder (vocab-size embedding-dim hidden-size)
  :parameters ((embedding (Embedding vocab-size embedding-dim :pad-idx 0))
               (layer     (RNN embedding-dim hidden-size :num-layers 1)))
  :forward ((x)
	    (with-calling-layers x
	      (embedding x)
	      (layer x))))

(defmodel Decoder (vocab-size embedding-dim hidden-size)
  :parameters ((embedding (Embedding vocab-size embedding-dim :pad-idx 0))
               (layer     (RNN embedding-dim hidden-size :num-layers 1))
	       (h2l       (linearlayer hidden-size vocab-size)))
  :forward ((encoder-state y)
	    (let* ((ye (call (self embedding) y))
		   (hs (call (self layer) ye encoder-state))
		   (h-output (call (self h2l) hs)))
	      (list h-output hs))))

(defmodel Seq2Seq (vocab-size embedding-dim input-size)
  :parameters ((encoder (Encoder vocab-size embedding-dim input-size))
	       (decoder (Decoder vocab-size embedding-dim input-size)))
  :forward ((x y)
	    (let ((x-state (call (self encoder) x))
		  (y1 (!zeros (!shape y))))
	      (setq y1 (setf (!aref y1 t) (!aref y '(1 0))))
	      (call (self decoder) x-state y1))))

(defun embedding-and-rnn-test ()
  (format t "~%Running Seq2Seq(RNN)~%")
  (let* ((model (Seq2Seq 2 16 10))
	 (x (!ones `(10 2)))
	 (y (!ones `(10 2)))
	 (out (time (call model x y))))
    (time (backward (!sum (car out))))
    t
    ))

(test networks-test
      (is (test-model linearlayer1 x))
      (is (test-model linearlayer2 x))
      (is (test-model denselayer1 x))
      (is (test-model denselayer2 x))
      (is (test-model denselayer3 x))
      (is (test-model denselayer4 x))
      (is (test-model dropout x)))

(test batchnorm-test
      (is (test-model batchnorm2d x)))

(test nlp-test
      (is (test-model embedding (parameter (!ones `(10 10)))))
      (is (test-model rnn1 words))
      (is (test-model rnn2 words))
      (is (embedding-and-rnn-test)))

(test model-list-test
      (is (test-model-list)))

