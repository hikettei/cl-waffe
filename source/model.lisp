
(in-package :cl-waffe)

(defun call (model &rest args)
  (let ((result (apply (slot-value model 'forward) model args)))
    (setf (slot-value result 'backward) (slot-value model 'backward))
    (setf (slot-value result 'grad) model)
    (setf (slot-value result 'variables) args)
    result))

(defmacro defmodel (name args &key parameters forward backward)
  (labels ((assure-args (x)
	     (if (or (equal (symbol-name x) "forward")
		     (equal (symbol-name x) "backward")
		     (equal (symbol-name x) "self")) ; enough?
		 (error "the name forward cant be used as param name")
		 x)))
    (unless forward
      (error "insufficient forms"))
    `(defmacro ,name (&rest init-args &aux (c (gensym)))
       `(progn
	  (defstruct (,(gensym (symbol-name ',name))
		     (:constructor ,c (,@',args &aux ,@',parameters)))
	    ,@',(map 'list (lambda (x) (assure-args (car x))) parameters)
	   (forward ,',(let ((largs (car forward))
			     (lbody (cdr forward))
			     (self-heap (gensym)))
			 `(dolist (i ,largs) (assure-args i))
			 `(lambda ,(concatenate 'list (list self-heap) largs)
			    (macrolet ((self (name)
					 `(slot-value ,',self-heap ',name)))
			      ,@lbody))))
	   (backward ,',(if backward
			    (let ((largs (car backward))
				  (lbody (cdr backward))
				  (self-heap (gensym)))
			      `(dolist (i ,largs) (assure-args i))
			      `(lambda ,(concatenate 'list (list self-heap) largs)
				 (macrolet ((self (name)
					      `(slot-value ,',self-heap ',name)))
				   ,@lbody)))
			    nil))) ;my grad
	 (,c ,@init-args)))))

