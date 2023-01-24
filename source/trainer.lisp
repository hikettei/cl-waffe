
(in-package :cl-waffe)

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
		      (zero-grad () `(unless *no-grad*
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

(defmacro deftrainer (name args &key model optimizer optimizer-args step-model predict (forward NIL) &aux (out-id (gensym)))
  (if forward (error ":forward is unavailable in deftrainer macro. use instead: :step-model"))
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
    
     (let ((constructor-name (gensym)))
	`(prog1
	  (defstruct (,name
		      (:print-function (lambda (trainer stream _)
					 (declare (ignore trainer _))
					 (format stream "[Trainer of ___]")))
		      (:constructor ,constructor-name (,@(map 'list (lambda (x) (assure-args x)) args)
						       &aux (model ,model)
							    (optimizer (cl-waffe.optimizers:init-optimizer ,optimizer
							   		              			    model
										         		    ,@optimizer-args)))))
		     (model NIL)
		     (optimizer NIL)
		     (optimizer-report t)
		     (predict    ,(if predict t nil))
		     (step-model ,(if step-model t nil)))
	   (define-trainer-method trainer-step-model ,name ,(car step-model) ,(cdr step-model))
	   (define-trainer-method trainer-predict    ,name ,(car predict)    ,(cdr predict))
	   (defun ,name (&rest init-args)
	     (apply #',constructor-name init-args))))))

(defun step-model (trainer &rest args)
  (uncheck-destructive args)
  (apply (trainer-step-model trainer) args))

(defun step-model1 (trainer args)
  (uncheck-destructive args)
  (apply (trainer-step-model trainer) args))

(defun predict (trainer &rest args)
  (uncheck-destructive args)
  (predict1 trainer args))

(defun predict1 (trainer args)
  (uncheck-destructive args)
  (apply (trainer-predict trainer) args))

(defmacro defdataset (name args &key parameters next length)
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
    
     (let ((constructor-name (gensym)))
       `(prog1
	    (defstruct (,name
			(:print-function (lambda (trainer stream _)
					   (declare (ignore trainer _))
					   (format stream "[Dataset of ___]")))
			(:constructor ,constructor-name (,@args &aux ,@parameters)))
	    ,@(map 'list (lambda (x) (assure-args (car x))) parameters)  
	    (length       t :type boolean)
	    (dataset-next t :type boolean))
	  (define-dataset-method dataset-next   ,name ,(car next)   ,(cdr next))
	  (define-dataset-method dataset-length ,name ,(car length) ,(cdr length))
	  (defun ,name (&rest init-args)
	    (apply #',constructor-name init-args))))))

(defun get-dataset (dataset index)
  (funcall (dataset-next dataset) index))

(defun get-dataset-length (dataset)
  (funcall (dataset-length dataset)))

(defun eq-ntimes (width &optional (word "="))
  (with-output-to-string (str) (dotimes (_ (* 2 width)) (format str word))))

(defun format-title (title start-from width)
  (let ((base (eq-ntimes width)))
    (setf (subseq base start-from (+ start-from (length title))) title)
    base))

(defmacro maxlist (list)
  `(let ((max-item (apply #'max ,list)))
     (if (<= max-item 1.0) 1.0 max-item)))

(defun max-position-column (arr)
  (declare (optimize (speed 3) (space 0) (safety 0) (debug 0))
           (type (array single-float) arr))
  (let ((max-arr (make-array (array-dimension arr 0)
                             :element-type 'single-float
                             :initial-element most-negative-single-float))
        (pos-arr (make-array (array-dimension arr 0)
                             :element-type 'fixnum
                             :initial-element 0)))
    (loop for i fixnum from 0 below (array-dimension arr 0) do
      (loop for j fixnum from 0 below (array-dimension arr 1) do
        (when (> (aref arr i j) (aref max-arr i))
          (setf (aref max-arr i) (aref arr i j)
                (aref pos-arr i) j))))
    pos-arr))

(defun valid (trainer dataset batch-size)
  (let ((count 0)
	(correct 0))
    (loop for index below (get-dataset-length dataset) by batch-size
	  do (let* ((ds (get-dataset dataset index))
		    (x (car ds))
		    (y (second ds))
		    (out (call (slot-value trainer 'model) x))
		    (out-labels (max-position-column (mgl-mat:mat-to-array (data out))))
		    (y-labels   (max-position-column (mgl-mat:mat-to-array (data y)))))
	       (loop for i below (length out-labels)
		     do (progn
			  (incf count 1)
			  (if (= (aref out-labels i) (aref y-labels i))
			      (incf correct 1)
			      (incf correct 0))))))
    (format t "Accuracy:~a%~C" (coerce (/ correct count) 'float) #\newline)))
			
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
				(height 10)) ; stream指定してtxtファイルにログを残せるようにしたい
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
      
      (loop for index below (/ total-len  batch-size)
	    do (let* ((i (if random (random (- total-len batch-size)) index))
		      (args (get-dataset dataset i))
		      (loss (data (step-model1 trainer args))))
		 (push loss losses)
		 (if (= (mod index 100) 0)
		     (cl-cram:update status-bar 0 :desc (format nil "~a/~a, loss:~a" index (/ total-len batch-size) (/ (apply #'+ losses) (length losses)))))))
      
      (format stream "~C" #\newline)
      (format stream "~C" #\newline)
      (cl-cram:update status-bar 1 :desc (format nil "loss:~a" (/ (apply #'+ losses) (length losses)))))

    (print "")
    (valid trainer valid-dataset batch-size)))

(defdataset WaffeDataset (train valid &key (batch-size 1))
  :parameters ((train train) (valid valid) (batch-size batch-size))
  :next    ((index)
	    (list (!set-batch (self train) index (self batch-size))
		  (!set-batch (self valid) index (self batch-size))))
  :length (() (car (!shape (self train)))))

