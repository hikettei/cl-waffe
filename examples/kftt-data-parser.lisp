
; The code below is made for utils.
; This package will be integrated with :cl-waffe.io
; So users don't have to implement by themselves
; These implementations are too slow

(defpackage :kftt-data-parser
  (:use :cl)
  (:export #:calc-max-length
	   #:calc-data-size
	   #:collect-tokens
	   #:init-train-datum
	   #:vector2word))

(in-package :kftt-data-parser)

(defparameter *data-path* "./kftt-data-1.0/data/")
(defparameter *file-begins-with* "/kyoto-")

(defmacro with-gensyms (syms &body body)
  `(let ,(mapcar #'(lambda (s)
                     `(,s (gensym)))
                 syms)
     ,@body))

(defmacro kftt-data-path (type data-type language)
  `(concatenate 'string
		*data-path*
		(case ,type
		  (:orig "orig")
		  (:tok "tok")
		  (T (error "Invaild keyform")))
		*file-begins-with*
		(case ,data-type
		  (:dev "dev")
		  (:test "test")
		  (:train "train")
		  (:tune "tune")
		  (:devtest "devtest")
		  (T (error "Invaild keyform")))
		"."
		(case ,language
		  (:en "en")
		  (:ja "ja")
		  (T (error "Invaild keyform")))))

(defun split (x str)
  (let ((pos (search x str))
        (size (length x)))
    (if pos
      (cons (subseq str 0 pos)
            (split x (subseq str (+ pos size))))
      (list str))))

(defun check-max (i maxlen)
  (if (eql maxlen T)
      T
      (< i maxlen)))

; Function for making dict
(defun collect-tokens (data-type language &optional (maxlen T) (w2i (make-hash-table :test 'equal)) (i2w (make-hash-table :test 'eq)))
  (register-word w2i i2w "<PAD>")
  (register-word w2i i2w "<UNK>")
  (register-word w2i i2w "<BOS>")
  (register-word w2i i2w "<EOS>")
  (let ((i 0))
    (with-open-file (f (kftt-data-path :tok data-type language) :external-format :utf8)
      (loop for line = (handler-case (read-line f nil nil)
			 (error (_) (declare (ignore _)) nil))
	    while line
	    do (if (check-max i maxlen)
		   (progn
		     (incf i 1)
		     (dolist (l (split " " line))
		       (register-word w2i i2w l))))))
    (values w2i i2w)))

(defun register-word (w2i i2w word)
  (let ((s (hash-table-count w2i)))
    (unless (gethash word w2i)
      (progn
	(setf (gethash word w2i) (1+ s))
	(setf (gethash (1+ s) i2w) word)))
    ; assure oneness
    (unless (= (hash-table-count w2i)
	       (hash-table-count i2w))
      (error "word count doesn't match"))))

(defmacro n2onehot (dict n)      
  `(if ,n
       ,n
       (gethash "<UNK>" ,dict)))


(defmacro onehot2n (onehot)
  `(position 1 ,onehot))

(defun word2vector (dict sentence max-length)
  (let ((translated-vector (make-array max-length))
	(tokenized (split " " sentence)))
    (dotimes (i (length tokenized))
      (setf (aref translated-vector i)
	    (n2onehot dict (gethash (nth i tokenized) dict))))
    (setf (aref translated-vector (length tokenized))
	  (n2onehot dict (gethash "<EOS>" dict)))
    translated-vector))

(defmacro vector2word (dict vector)
  (with-gensyms (translated-word i)
    `(let ((,translated-word (make-array (length ,vector))))
       (dotimes (,i (length ,vector))
	 (setf (aref ,translated-word ,i)
	       (gethash (nth ,i ,vector) ,dict)))
       ,translated-word)))

(defmacro with-open-kftt-file (line type data-type language &body body)
  (with-gensyms (buffer)
    `(with-open-file (,buffer (kftt-data-path ,type ,data-type ,language) :external-format :utf8)
       (loop for ,line = (handler-case (read-line ,buffer nil nil)
			   (error (_) (declare (ignore _)) nil))
	     while ,line
	     do ,@body))))

(defun translate-into-vector (target data-type lang dict max-length ds)
  (let ((i 0))
    (with-open-kftt-file line :tok data-type lang
      (if (< i ds)
	  (progn
	    (setf (aref target i)
		  (word2vector dict line max-length))
	    (incf i 1))))))

(defun list-to-2d-array (list)
  (make-array (list (length list)
                    (length (aref list 0)))
              :initial-contents (coerce (map 'list (lambda (x)
						     (coerce x 'list))
					     list)
					'list)))

(defun init-train-datum  (data-type
			  lang1
			  lang2
			  w2i-lang1
			  w2i-lang2
			  &optional
			  (data-size (calc-data-size :tok data-type lang1)))
  (let* ((max-length (max (calc-max-length :tok data-type lang1)
			  (calc-max-length :tok data-type lang2)))
	 (train-x (make-array data-size))
	 (train-y (make-array data-size)))
    
    (translate-into-vector train-x data-type lang1 w2i-lang1 max-length data-size)
    (translate-into-vector train-y data-type lang2 w2i-lang2 max-length data-size)
    (values (list-to-2d-array train-x)
	    (list-to-2d-array train-y))))


(defun calc-max-length (type data-type lang)
  (let ((max-size 0))
    (with-open-kftt-file line type data-type lang
      (setq max-size (max max-size (length (split " " line)))))
    ; In the end of sequence, we allocate the place for <EOS>
    (1+ max-size)))

(defun calc-data-size (type data-type lang)
  (let ((data-size 0))
    (with-open-kftt-file _ type data-type lang
      (incf data-size 1))
    data-size))
