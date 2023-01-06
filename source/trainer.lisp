
(in-package :cl-waffe)

(defmacro deftrainer (name args &key model optimizer optimizer-args step-model (forward NIL))
  (if forward (error ":forward is unavailable in deftrainer macro. use instead: :step-model"))
  (labels ((assure-args (x)
		     (if (or (eq (symbol-name x) "model")
			     (eq (symbol-name x) "optimizer")
			     (eq (symbol-name x) "step"))
			 (error "cant use ~a as a name" (symbol-name x))
			 x)))
     (let ((constructor-name (gensym)))
	`(prog1
	  (defstruct (,name
		      (:print-function (lambda (trainer stream _)
					 (declare (ignore trainer _))
					 (format stream "[Trainer of ___]")))
		      (:constructor ,constructor-name (,@(map 'list (lambda (x) (assure-args x)) args)
						       &aux (model (funcall (lambda () ,model)))
							 (optimizer (cl-waffe.optimizers:init-optimizer ,optimizer
													model
													,@optimizer-args)))))
		     (model NIL)
		     (optimizer NIL)
		     (step-model ,(let ((largs (car step-model))
					  (lbody (cdr step-model))
					  (self-heap (gensym)))
				      `(lambda ,(concatenate 'list (list self-heap) largs)
					 (macrolet ((model     ()            `(slot-value ,',self-heap 'model))
						    (update    (&rest args1) `(call (slot-value ,',self-heap 'optimizer) ,@args1))
						    (zero-grad ()            `(funcall (slot-value (slot-value ,',self-heap 'optimizer) 'backward)
										       (slot-value ,',self-heap 'optimizer)
										       (slot-value ,',self-heap 'model))))
					   ,@lbody)))))
	   (defun ,name (&rest init-args)
	     (apply #',constructor-name init-args))))))

(defun step-model (trainer &rest args)
  (apply (slot-value trainer 'step-model) trainer args))

(defun step-model1 (trainer args)
  (apply (slot-value trainer 'step-model) trainer args))

(defmacro defdataset (name args &key parameters forward length)
  (labels ((assure-args (x)
		     (if (or (eq (symbol-name x) "parameters")
			     (eq (symbol-name x) "forward")
			     (eq (symbol-name x) "length"))
			 (error "cant use ~a as a name" (symbol-name x))
			 x)))
    (unless forward
      (error ""))
    (unless length
      (error ""))
     (let ((constructor-name (gensym)))
       `(prog1
	    (defstruct (,name
			(:print-function (lambda (trainer stream _)
					   (declare (ignore trainer _))
					   (format stream "[Dataset of ___]")))
			(:constructor ,constructor-name (,@args &aux ,@parameters)))
	    ,@(map 'list (lambda (x) (assure-args (car x))) parameters)
	    (length ,(let ((largs (car length))
			   (lbody (cdr length))
			   (self-heap (gensym)))
		       `(lambda ,(concatenate 'list (list self-heap) largs)
			  (macrolet ((self (name) `(slot-value ,',self-heap ',name)))
			    ,@lbody))))
	    (forward ,(let ((largs (car forward))
			    (lbody (cdr forward))
			    (self-heap (gensym)))
			`(lambda ,(concatenate 'list (list self-heap) largs)
			   (macrolet ((self (name) `(slot-value ,',self-heap ',name)))
			     ,@lbody)))))
	  (defun ,name (&rest init-args)
	    (apply #',constructor-name init-args))))))

(defun get-dataset (dataset index)
  (funcall (slot-value dataset 'forward) dataset index))

(defun get-dataset-length (dataset)
  (apply (slot-value dataset 'length) (list dataset)))

(defun eq-ntimes (width &optional (word "="))
  (with-output-to-string (str) (dotimes (_ (* 2 width)) (format str word))))

(defun format-title (title start-from width)
  (let ((base (eq-ntimes width)))
    (setf (subseq base start-from (+ start-from (length title))) title)
    base))

(defmacro maxlist (list)
  `(let ((max-item (apply #'max ,list)))
     (if (<= max-item 1.0) 1.0 max-item)))


(defun train (trainer dataset &key (enable-animation t)
		                (epoch 1)
			     	(max-iterate nil)
				(verbose t)
				(stream t)
				(progress-bar-freq 1)
				(save-model-path nil)
				(width 45)
				(height 10)) ; stream指定してtxtファイルにログを残せるようにしたい
  (let ((losses `(0.0)) ; cl-termgraph assumes that loss >= 0
	(prev-losses nil)
	(prev-losses1 nil)
	(status-bar nil))
    (if (and enable-animation verbose)
	(cl-cram:init-progress-bar status-bar (format nil "loss:~a" (first losses)) (get-dataset-length dataset)))
    (dotimes (epoch-num epoch)
      (if verbose
	  (progn
	    (format stream "~C~a~C" #\newline (eq-ntimes width "–") #\newline)
	    (format stream "~C~a~C" #\newline  (format-title (format nil "|~a Epoch:|" epoch-num) 4 width) #\newline)
	    (format stream "~C~a~C" #\newline (eq-ntimes width "–") #\newline)))

      (let ((total-len (if max-iterate max-iterate (get-dataset-length dataset))))
	(setq cl-termgraph:*dif* (if (< width total-len) (* width (/ 1 total-len)) (/ 1 total-len)))
	(dotimes (i total-len)
	  (let* ((args (get-dataset dataset i))
		 (loss (data (step-model1 trainer args))))
	    (push loss losses)
	    (if (and enable-animation verbose)
		(cl-cram:update status-bar 1 :desc (format nil "loss:~a" (first losses))))))
	(let* ((losses-aorder (map 'list (lambda (x) (* (/ x (maxlist (butlast losses))) (1+ height))) (cdr (reverse losses))))
	       (pallet (cl-termgraph:make-listplot-frame (* 2 width) height)))
	  ;(cl-termgraph:init-line pallet :white)
	  ;(cl-termgraph:listplot-write pallet losses-aorder :blue)
	  ;(if prev-losses
	  ;    (cl-termgraph:listplot-write pallet prev-losses :red))
	  ;(format stream "~C" #\newline)
	  ;(cl-termgraph:listplot-print pallet :x-label "n" :y-label "loss" :title nil
		;			      :descriptions (if prev-losses
		;						`((:red "prev-losses" ,(apply #'min prev-losses1) ,(apply #'max prev-losses1))
		;						  (:blue "losses" ,(apply #'min losses) ,(apply #'max losses)))
		;						`((:blue "losses" ,(apply #'min losses) ,(apply #'max losses))))
		;			      :stream stream)
	  (setq prev-losses1 losses)
	  (setq prev-losses losses-aorder)
	  (cl-cram:update status-bar 0 :desc (format nil "Preparing for Next Batch...") :reset t)
	  (setq losses `(0.0)))))))


