
(in-package :cl-waffe)


(defmacro deftrainer (name args &key model optimizer optimizer-args step-model (forward NIL))
  (if forward (error ":forward is unavailable in deftrainer macro. use instead: :step-model"))
  (labels ((assure-args (x)
		     (if (or (eq (symbol-name x) "model")
			     (eq (symbol-name x) "optimizer")
			     (eq (symbol-name x) "step"))
			 (error "cant use ~a as a name" (symbol-name x))
			 x)))
     `(defmacro ,name (&rest init-args &aux (constructor-name (gensym)))
	`(progn
	  (defstruct (,(gensym (symbol-name ',name))
		      (:print-function (lambda (trainer stream _)
					 (declare (ignore trainer _))
					 (format stream "[Trainer of ___]")))
		      (:constructor ,constructor-name (,@',(map 'list (lambda (x) (assure-args x)) args)
						       &aux (model ,',model)
							 (optimizer (cl-waffe.optimizers:init-optimizer ,',optimizer
													model
													,@',optimizer-args)))))
		     (model NIL)
		     (optimizer NIL)
		     (step-model ,',(let ((largs (car step-model))
					  (lbody (cdr step-model))
					  (self-heap (gensym)))
				      `(lambda ,(concatenate 'list (list self-heap) largs)
					 (macrolet ((model     ()            `(slot-value ,',self-heap 'model))
						    (update    (&rest args1) `(call (slot-value ,',self-heap 'optimizer) ,@args1))
						    (zero-grad ()            `(funcall (slot-value (slot-value ,',self-heap 'optimizer) 'backward)
										       (slot-value ,',self-heap 'optimizer))))
					    ,@lbody)))))
	  (,constructor-name ,@init-args)))))


(defun step-model (trainer &rest args)
  (apply (slot-value trainer 'step-model) trainer args))

