
(in-package :cl-waffe)

(defmacro call (model &rest args)
  `(funcall (slot-value ,model 'forward) ,model ,@args))

(defmacro defmodel (name &key args parameters forward)
  (labels ((assure-args (x)
	     (if (or (equal (symbol-name x) "forward")
		     (equal (symbol-name x) "self")) ; enough?
		 (error "the name forward cant be used as param name")
		 x)))
    (unless (and parameters forward)
      (error "insufficient params"))
    `(defmacro ,name (&rest init-args &aux (c (gensym)))
       `(progn
	 (defstruct (,(gensym (symbol-name ',name))
		     (:constructor ,c (,',@args &aux ,@',parameters)))
	   ,@',(map 'list (lambda (x) (assure-args (car x))) parameters)
	   (forward ,',(let ((largs (car forward))
			     (lbody (cdr forward))
			     (self-heap (gensym)))
			 `(dolist (i ,largs) (assure-args i))
			 `(lambda ,(concatenate 'list (list self-heap) largs)
			    (macrolet ((self (name)
					 `(slot-value ,',self-heap ',name)))
			      ,@lbody)))))
	 (,c ,@init-args)))))

