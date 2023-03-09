
(in-package :cl-waffe)

#|
Here's utils for deftrainer/defdataset
and train/valid function.
|#
(defun uncheck-destructive (variables)
  (dolist (v variables)
    (if (typep v 'waffetensor)
	(setf (waffetensor-destructive? v) nil))))

(defgeneric trainer-step-model (trainer))
(defgeneric trainer-predict   (trainer))

(defgeneric dataset-next      (trainer))
(defgeneric dataset-length    (trainer))

(defmacro define-trainer-method (fname name args body)
  (let ((f-ident   (gensym (symbol-name name)))
	(self-heap (gensym (symbol-name name))))
    `(progn
	 (defun ,f-ident (,self-heap ,@args)
	   (macrolet ((self (name) `(slot-value ,',self-heap ',name))
		      (model () `(self model))
		      (update (&rest args1) `(unless *no-grad*
				                 (with-no-grad (funcall (call-forward (self optimizer)) ,@args1))))
		      (zero-grad ()
			`(unless *no-grad*
			   (funcall (call-backward (self optimizer)) (self model)))))
	     ,@body))
	 (defmethod ,fname ((self ,name))
	   (lambda (&rest node-inputs) (apply #',f-ident self node-inputs))))))

(defmacro define-dataset-method (fname name args body)
  (let ((f-ident   (gensym (symbol-name name)))
	(self-heap (gensym (symbol-name name))))
    `(progn
	 (defun ,f-ident (,self-heap ,@args)
	   (macrolet ((self (name) `(slot-value ,',self-heap ',name)))
	     ,@body))
	 (defmethod ,fname ((self ,name))
	   (lambda (&rest node-inputs) (apply #',f-ident self node-inputs))))))

; This model used for just only making thread.
(defmodel TrainerInputLayer (caller)
  :optimize t
  :parameters ((caller caller :type function))
  :forward (()
	    (funcall (self caller))))

(defmacro deftrainer (name args
		      &key
			model
			optimizer
			optimizer-args
			step-model
			predict
			(document "An trainer structure defined by cl-waffe."))
  "Defining trainer, which is made in order to call @cl:param(train) function.

The slots you defined can be invoked by using @c((step-model model &rest args)), @c((predict model &rest args)). See below.


@begin(deflist)

@term(model)
@def(An model defined by @c((defmodel)) which you want to train.)

@term(optimizer)
@def(An optimizer defined by @c((defoptimizer)))

@term(optimizer-args)
@def(An arguments for optimizer)

@term(step-model)
@def(For each batch step, :step-model is called in @c((train)) function. Describe here forward step, backward, zero-grad, update for training.)

@term(predict)
@def(an code for predicting)

@end(deflist)


These macro below are defined by @cl:spec(macrolet) and you can use them in :step-model, :predict

@begin(deflist)
@term((self name))
@def(access trainer's parameters.)

@term((model))
@def(access trainer's model, defined by :model keyword.)

@term((zero-grad))
@def(Find model's all parameters and constants, and initialize their grads. (i.e. call optimizer's backward))

@term((update))
@def(Find model's all parameters, and call optimizer and change parameter's data. (i.e. call optimizer's forward))
@end(deflist)

This trainer macro is defined in order to integrate following works:

@begin(enum)
  @item(calling models)
  @item(calling criterions)
  @item(calling backward)
  @item(calling optimizer)
  @item(calling zero-grad)
  @item(defining predict)
@end(enum)

Example:

@begin[lang=lisp](code)
(deftrainer MLPTrainer (activation lr)
  :model          (MLP activation)
  :optimizer      cl-waffe.optimizers:Adam ; Note: :optimizer requires a single variable.
  :optimizer-args (:lr lr) ; these arguments directly expanded to optimizer's args.
  :step-model ((x y)
	       (zero-grad) ; call zero-grad
	       (let ((out (cl-waffe.nn:softmax-cross-entropy (call (model) x) y))) ; get criterion
		 (backward out) ; backward
		 (update) ; call optimizer
		 out)) ; return loss
 :predict ((x) (call (model) x))) ;for predict

(setq trainer (MLPTrainer :relu 1e-4)) ; init your trainer

; Train:   (step-model trainer model-input-x model-input-y)
; Predict: (predict trainer model-input-x)

@end[lang=lisp](code)"
  (labels ((assure-args (x)
	     (if (or (eq (symbol-name x) "model")
		     (eq (symbol-name x) "predict")
		     (eq (symbol-name x) "optimizer")
		     (eq (symbol-name x) "optimizer-report")
		     (eq (symbol-name x) "step-model"))
		 (error "cant use ~a as a name" (symbol-name x))
		 x)))
     (unless step-model
       (error "deftrainer: the slot :step-model is nil. Please fill here"))
    (unless predict
       (error "deftrainer: the slot :predict is nil. Please fill here"))
    
     (progn
	`(progn
	  (defstruct (,name
		      (:print-function (lambda (trainer stream _)
					 (declare (ignore trainer _))
					 (format stream "<Dataset: ~a()>"
						 (symbol-name ',name))))
		      (:constructor ,name (,@(map 'list (lambda (x) (assure-args x)) args)
						       &aux (model ,model)
							    (optimizer (cl-waffe.optimizers:init-optimizer ,optimizer
							   		              			    model
										         		    ,@optimizer-args)))))
	             ,document
		     (model NIL)
		     (optimizer NIL)
		     (optimizer-report t)
		     (predict    ,(if predict t nil))
		     (step-model ,(if step-model t nil)))
	   (define-trainer-method trainer-step-model ,name ,(car step-model) ,(cdr step-model))
	   (define-trainer-method trainer-predict    ,name ,(car predict)    ,(cdr predict))
	   nil))))

(defun step-model (trainer &rest args)
  "An function for calling trainer object defined by deftrainer
By using this function, trainer's step-model will be invoked.

Input: Trainer, Args"
  (step-model1 trainer args))

(defun step-model1 (trainer args)
  "The same as step-model but args must be cons"
  (uncheck-destructive args)
  (labels ((forward ()
	     (apply (trainer-step-model trainer) args)))
    (call (TrainerInputLayer #'forward))))

(defun predict (trainer &rest args)
  "An function for calling trainer's predict slot"
  (predict1 trainer args))

(defun predict1 (trainer args)
  "The same as predict1 byt args must be cons"
  (uncheck-destructive args)
  (labels ((predict-lambda ()
	     (apply (trainer-predict trainer) args)))
    (call (TrainerInputLayer #'predict-lambda))))

(defun print-dataset (trainer stream _)
  (declare (ignore trainer _))
  (format stream "[Dataset of ___]"))

(defmacro defdataset (name args &key parameters next length (document "An dataset structure defined by cl-waffe."))
  "Defining dataset. (This is kinda pytorch's dataloader)

The slots you defined can be invoked by using (get-dataset dataset index) (get-length dataset).

@begin(deflist)
@term(parameters)
@def(parameters datasets have.)

@term(next)
@def(when function (get-dataset dataset index) is called, this slot invokes. Return waffetensor for the next batch in response to your task.)

@term(length)
@def(In this form, the function must return the total length of your datasets where the value is fixnum. (Not a batch, and not a current index.))

@end(deflist)

@begin[lang=lisp](code)

(defdataset Mnistdata (train valid batch-size)
  :parameters ((train train) (valid valid) (batch-size batch-size))
  :next    ((index)
	    (list (!set-batch (self train) index (self batch-size))
		  (!set-batch (self valid) index (self batch-size))))
  :length (() (car (!shape (self train)))))

@end[lang=lisp](code)

cl-waffe excepts index to be 1, 2, 3, ... (dataset-maxlen)

So, please manage batch-sizes in args and :next slots."
  (labels ((assure-args (x)
		     (if (or (eq (symbol-name x) "parameters")
			     (eq (symbol-name x) "next")
			     (eq (symbol-name x) "length"))
			 (error "cant use ~a as a name" (symbol-name x))
			 x)))
    (unless next
      (error "defdataset: the slot :next is nil. Please fill here the code returning next batch data"))
    (unless length
      (error "defdataset: the slot :length is nil. Please fill here the code returning the total size of a training data."))
    
    (let* ((document (eval document))
	   (doc-output (typecase document
			 (string document)
			 (waffeobjectusage
			  (build-docstring document :dataset))
			 (T "None"))))
       `(progn
	    (defstruct (,name
			(:print-function print-dataset)
			(:constructor ,name (,@args &aux ,@(map 'list (lambda (x) `(,(car x) ,(second x))) parameters))))
	      ,doc-output
	    ,@(map 'list (lambda (x) `(,(assure-args (car x)) ,(second x) ,@(cddr x))) parameters)
	    (length       t :type boolean)
	    (dataset-next t :type boolean))
	  (define-dataset-method dataset-next   ,name ,(car next)   ,(cdr next))
	  (define-dataset-method dataset-length ,name ,(car length) ,(cdr length))
	  nil))))

(defun get-dataset (dataset index)
  "Get datum of the index from dataset.
Input: dataset ... dataset defined by defdataset.
       index ... fixnum"
  (funcall (dataset-next dataset) index))

(defun get-dataset-length (dataset)
  "Get total size of your dataset."
  (funcall (dataset-length dataset)))

(defun eq-ntimes (width &optional (word "="))
  (with-output-to-string (str) (dotimes (_ (* 2 width)) (format str word))))

(defun format-title (title start-from width)
  (let ((base (eq-ntimes width)))
    (setf (subseq base start-from (+ start-from (length title))) title)
    base))

(defun valid (trainer dataset batch-size)
  "Valid trainer"
  (let ((count 0)
	(correct 0))
    (loop for index fixnum upfrom 0 below (get-dataset-length dataset) by batch-size
	  do (let* ((ds (get-dataset dataset index))
		    (x (car ds))
		    (y (second ds))
		    (out (const (value (call (slot-value trainer 'model) x))))
		    (out-labels (!argmax out))
		    (y-labels   (!argmax (const (copy-mat (data y))))))
	       (with-facets ((out-labels ((data out-labels) 'backing-array :direction :input))
			     (y-labels   ((data y-labels) 'backing-array :direction :input)))
		 (loop for i below (length out-labels)
		       do (progn
			    (incf count 1)
			    (if (= (aref out-labels i) (aref y-labels i))
				(incf correct 1)
				(incf correct 0)))))))
	     (format t "Accuracy:~a~C" (coerce (/ correct count) 'float) #\newline)))
  
(defun train (trainer dataset &key (valid-dataset nil)
				(valid-each 100)
				(enable-animation t)
		                (epoch 1)
				(batch-size 1)
			     	(max-iterate nil)
				(verbose t)
				(stream t)
				(progress-bar-freq 1)
				(save-model-path nil)
				(width 45)
				(random nil)
				(height 10)
				(print-each 10)) ; stream指定したい/cl-gnuplot/cl-termgraph使う
  "Trainining given trainer. If any, valid @cl:param(valid-dataset)

@begin(deflist)
@term(trainer)
@def(Trainer you defined by deftrainer)

@term(dataset)
@def(Dataset you defined by defdataset)

@term(valid-dataset)
@def(If valid-dataset=your dataset, use this to valid. If nil, ignored)

@term(enable-animation)
@def(Ignored)

@term(epoch)
@def(Iterate training by epoch, default=1)

@term(batch-size)
@def(Do batch training. default=1)

@term(verbose)
@def(if t, put log to stream)

@end(deflist)

This function is temporary and other arguments are ignored.

And this function has a lot of todo."
  (declare (ignore valid-each max-iterate progress-bar-freq save-model-path height))
  (let ((losses nil) ; cl-termgraph assumes that loss >= 0
	(status-bar nil)
	(total-len (get-dataset-length dataset)))
    (if (and enable-animation verbose)
	(cl-cram:init-progress-bar status-bar (format nil "loss:~a" (first losses)) epoch))
    (dotimes (epoch-num epoch)
      (setq losses nil)
      (if verbose
	  (progn
	    (format stream "~C~a~C" #\newline (eq-ntimes width "–") #\newline)
	    (format stream "~C~a~C" #\newline (format-title (format nil "|~a Epoch:|" epoch-num) 4 width) #\newline)
	    (format stream "~C~a~C" #\newline (eq-ntimes width "–") #\newline)))
      
      (loop for index fixnum below (/ total-len  batch-size)
	    do (let* ((i (if random (random (- total-len batch-size)) index))
		      (args (get-dataset dataset i))
		      (loss (data (step-model1 trainer args))))
		 (push loss losses)
		 (if (= (mod index print-each) 0)
		     (cl-cram:update status-bar 0 :desc (format nil "~a/~a, loss:~a" index (/ total-len batch-size) (/ (apply #'+ losses) (length losses)))))))
      
      (format stream "~C" #\newline)
      (format stream "~C" #\newline)
      (cl-cram:update status-bar 1 :desc (format nil "loss:~a" (/ (apply #'+ losses) (length losses)))))

    (fresh-line)
    (if valid-dataset
	(valid trainer valid-dataset batch-size))))

(defdataset WaffeDataset (train valid &key (batch-size 1))
  :document (with-usage "WaffeDataSet"
	      :overview "The standard dataset for 2d training data."
	      :args "train valid &key (batch-size 1)")
  :parameters ((train
		train
		:type
		waffetensor)
	       (valid
		valid
		:type
		waffetensor)
	       (batch-size
		batch-size
		:type
		fixnum))
  :next    ((index)
	    (list (!set-batch (self train) index (self batch-size))
		  (!set-batch (self valid) index (self batch-size))))
  :length (() (car (!shape (self train)))))

