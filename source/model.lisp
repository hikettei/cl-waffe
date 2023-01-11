
(in-package :cl-waffe)

(declaim (inline call))

(defparameter *no-grad* nil)

(defgeneric call-forward  (self))
(defgeneric call-backward (self))

(defmacro with-no-grad (&body body)
  `(progn
     (setq *no-grad* t)
     ,@body
     (setq *no-grad* nil)
     nil))

(declaim (inline call))
(defun call (model &rest args)
  ; calculating op(x,y) -> result(x, y), state
  (let* ((result (apply (call-forward model) args)))
    (if (slot-value model 'hide-from-tree) ;is a result model or not?, then is it the part of node?
	(unless *no-grad*
	  (setf (waffetensor-backward result) t)
	  (setf (waffetensor-state result) model)
	  (setf (waffetensor-variables result) args)
	  (setf (waffetensor-is-ancestor-param result) (if (member-if #'(lambda (x)
									  (waffetensor-is-ancestor-param x))
								      args)
							   t
							   nil))
	  result)
	result)))

(defmacro is-waffe-model (model)
  `(and (slot-exists-p ,model 'parameters)
        (slot-exists-p ,model 'hide-from-tree)
        (slot-exists-p ,model 'forward)
        (slot-exists-p ,model 'backward)))

(defun find-variables (model)
  (let ((parameters `(T)))
    (labels ((search-param (m)
	       (if (is-waffe-model m)
		   (dolist (p (slot-value m 'parameters))
		     (search-param (slot-value m p)))
		   (if (typep m 'WaffeTensor)
		       (push m parameters)))))
      (search-param model)
      (if (= (length parameters) 1)
	  (error "Could not find any parameter")
	  (butlast parameters)))))

(defmacro defoptimizer (name args &key parameters update)
  `(progn
     (defmodel ,name ,args
       :parameters ,parameters
       :forward ,update
       :backward ((model) (dolist (p (find-variables model))
			    (setf (waffetensor-state p) nil)
			    (setf (waffetensor-backward p) nil)
			    (setf (waffetensor-variables p) nil)
			    (setf (waffetensor-grad p) `(nil nil))
			    (let ((grad-tmp (waffetensor-grad-tmp p)))
			      (setf (grad-tmp-grad-called grad-tmp) nil)
			      (if (typep (grad-tmp-value grad-tmp) 'mgl-mat:mat)
				  (setf (grad-tmo-value grad-tmp) nil);(mgl-mat:fill! (grad-tmp-value grad-tmp) 0)
				  (setf (grad-tmp-value grad-tmp) nil))))
		nil)
     :hide-from-tree nil)))

(defmacro defnode (name args &key parameters forward backward)
  `(defmodel ,name ,args :parameters ,parameters :forward ,forward :backward ,backward :hide-from-tree T))

(defun check-destructive (variables)
  (dolist (v variables)
    (if (typep v 'waffetensor)
	(setf (waffetensor-destructive? v) nil))))
     
(defmacro define-node-method (fname name args body)
  (let ((f-ident   (gensym (symbol-name name)))
	(self-heap (gensym (symbol-name name))))
    `(progn
	 (defun ,f-ident (,self-heap ,@args)
	   ;(declare (type waffetensor ,@args))
	   (macrolet ((self (name) `(slot-value ,',self-heap ',name)))
	     ,@body))
       (defmethod ,fname ((self ,name))
	 (lambda (&rest node-inputs) (apply #',f-ident self node-inputs))))))

(defmacro defmodel (name args &key parameters forward backward hide-from-tree)
  #|
  Define an node.
  Args: hide-from-tree ... if true, this node is detached from autograd. (that is, when backward, use backward defined in itself)
  |#
  (labels ((assure-args (x)
	     (declare (type symbol x))
	     (if (or (equal (symbol-name x) "forward")
		     (equal (symbol-name x) "backward")
		     (equal (symbol-name x) "hide-from-tree")
		     (equal (symbol-name x) "parameters")
		     (equal (symbol-name x) "self"))
		 (error "the name ~a is not allowed to use" (symbol-name x))
		 x)))
    (unless forward
      (error ":forward slot is need to be fulfilled. When defining Model [~a]" name))
    
    (let ((constructor-name (gensym)))
      `(prog1
	   (defstruct (,name
		       (:print-function (lambda (m stream k)
					  (declare (ignore k))
					  (render-simple-model-structure stream m)))
		       (:constructor ,constructor-name (,@args &aux ,@parameters)))
	     (hide-from-tree ,hide-from-tree :type boolean)
	     (forward t :type boolean)
	     (backward ,(if backward t nil) :type boolean)
	     (parameters ',(map 'list (lambda (x) (assure-args (car x))) parameters))
	     ,@(map 'list (lambda (x) (assure-args (car x))) parameters))
	   (define-node-method call-forward  ,name ,(car forward)  ,(cdr forward))
	   (define-node-method call-backward ,name ,(car backward) ,(cdr backward))
	   (defun ,name (&rest init-args)
	     (apply #',constructor-name init-args))))))


(defun render-simple-model-structure (stream model)
  (format stream "[~a: ~a]" (if (slot-value model 'hide-from-tree)
				"Node "
				"Model")
	  (type-of model)))

(defun print-model (model)
  (fresh-line)
  (render-model-structure t model))

(defparameter *initial-indent-size* 4)

(defun render-model-structure (stream model &optional (indent-level 0) (total-param 0) (model-name "Model") (indent-increase 4) (prefix "Param"))
  (if (and (slot-exists-p model 'parameters)
	   (slot-exists-p model 'hide-from-tree)
	   (slot-exists-p model 'forward)
	   (slot-exists-p model 'backward))
      (if (typep (slot-value model 'parameters) 'list)
	  (labels ((indent-with (code &optional (space NIL))
		     (dotimes (_ (+ indent-level (if space *initial-indent-size* 0)))
		       (format stream code))))
	    (indent-with "–")
	    (format stream "––– {~a (~a)}~C" model-name (type-of model) #\newline)
	    (indent-with " " T)
	    (format stream "Input Shape : ( None )~C" #\newline)
	    (indent-with " " T)
	    (format stream "Output Shape: ( None )~C" #\newline)
	    (indent-with " " T)
	    (format stream "Params      : ( None )~C" #\newline)
	    (dotimes (i (length (slot-value model 'parameters)))
	      (let ((p (case i
			 (0 "Input ")
			 ;((1- (length (slot-value model 'parameters))) "Param ")
			 (T "Param ")))
		    (param (nth i (slot-value model 'parameters))))
		(render-model-structure stream (slot-value model param) (+ indent-level indent-increase) total-param (symbol-name param) indent-increase p)))

	    (if (= indent-level 0)
		(progn
		  (format stream ""))))) ;in the end of model
      (labels ((indent-with ()
		 (dotimes (_ (+ indent-level *initial-indent-size*))
		   (format stream " "))))
	(if (typep model 'WaffeTensor)
	    (progn
	      (indent-with)
	      (format stream "(+)~a:~a~C" prefix (!shape model) #\newline))))))


