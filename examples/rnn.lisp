
(defpackage :rnn-example
  (:use :cl :cl-waffe :cl-waffe.nn :kftt-data-parser)
  (:export demo))

(in-package :rnn-example)

; Todo: BatchNorm/Dropout

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
	      (setq y1 (setf (!aref y1) (!aref y '(1 0))))
	      (call (self decoder) x-state y1))))

(deftrainer Seq2SeqTrainer (vocab-size embedding-dim hidden-size)
  :model (Seq2Seq vocab-size embedding-dim hidden-size)
  :optimizer cl-waffe.optimizers:Adam
  :optimizer (:lr 1e-4)
  :step-model ((x y)
	       (zero-grad)
	       (let* ((outs (call (self model) x y))
		      (out (softmax-cross-entropy (car outs) y)))
		 (backward out)
		 (update)
		 out))
  :predict ((x) (call (model) x)))


; Todo 3d operations, for matmul and sum without using !aref

(defun demo (&key
	       (lang1 :ja)
	       (lang2 :en)
	       (maxlen 30)
	       (batch-size 16)
	       (embedding-dim 64)
	       (hidden-dim 128))

  ; Loadig Dataset

  ; Create dict
  (multiple-value-bind (lang1-w2i lang1-i2w)
      (collect-tokens :train lang1 maxlen)
    (defparameter w2i lang1-w2i)
    (defparameter i2w lang1-i2w))
  

  (multiple-value-bind (lang2-w2i lang2-i2w)
      (collect-tokens :test lang2 maxlen w2i i2w)
    (setq w2i lang2-w2i)
    (setq i2w lang2-i2w))
  

  (format t "==The dictionary size is ~a~%" (hash-table-count w2i))

  (format t "==The total size of training dataset is ~a~%" (calc-data-size :tok :train lang1))

  
  (multiple-value-bind (l1 l2) (init-train-datum :train lang1 lang2
						 w2i w2i maxlen)
    (defparameter lang1-train (const l1))
    (defparameter lang2-train (const l2)))
  
  (multiple-value-bind (l1 l2) (init-train-datum :test lang1 lang2
						 w2i w2i)
    (defparameter lang1-test (const l1))
    (defparameter lang2-test (const l2)))

  (print "Lang1 Train")
  (print lang1-train)
  (print "Lagn2 Train")
  (print lang2-train)

  (print "Lang1 Test")
  (print lang1-test)
  (print "Lang2 Test")
  (print lang2-test)

  ; Initilizing datasetss
  (defparameter dataset-train (WaffeDataset lang1-train lang2-train
					    :batch-size batch-size))
  (defparameter dataset-valid (WaffeDataset lang1-test lang2-test
					    :batch-size batch-size))

  (defparameter vocab-size (hash-table-count w2i))
  (defparameter model (Seq2SeqTrainer vocab-size embedding-dim hidden-dim))

  (train model
	 dataset-train
	 :epoch 10
	 :batch-size batch-size
;	 :valid-dataset dataset-valid
	 :print-each 100
	 :verbose t
	 :random t))
