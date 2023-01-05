
(in-package :cl-waffe)


(defun call (model &rest args)  
  (let ((result (apply (slot-value model 'forward) model args)))
    (if (slot-value model 'hide-from-tree) ;assure model isnt model
	(progn
	  (setf (waffetensor-backward result) (slot-value model 'backward))
	  (setf (waffetensor-state result) model) ; last state
	  (setf (waffetensor-variables result) (coerce args 'list))
	  result)
	result)))

(defun call1 (model &rest args)
  (let ((result (apply #'.call model args)))
    (if (typep (data result) 'mgl-mat:mat)
	(if (equal (mgl-mat:mat-dimensions (data result)) `(1))
	    (!1darray-to-const result)
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

(defmacro defoptimizer (name args &key parameters update &aux (model (gensym)))
  `(defmodel ,name ,args
     :parameters ,parameters
     :forward ,update
     :backward ((,model) (dolist (p (find-variables ,model))
			   (setf (waffetensor-state p) nil)
			   (setf (waffetensor-backward p) nil)
			   (setf (waffetensor-variables p) nil)
			   (setf (waffetensor-grad p) `(nil nil))
			   (let ((grad-tmp (waffetensor-grad-tmp p)))
			     (setf (grad-tmp-grad-called grad-tmp) nil)
			     (if (typep (grad-tmp-value grad-tmp) 'mgl-mat:mat)
				 (mgl-mat:fill! (grad-tmp-value grad-tmp) 0)
				 (setf (grad-tmp-value grad-tmp) nil))))
		nil)
     :hide-from-tree nil))

(defmacro defnode (name args &key parameters forward backward)
  `(defmodel ,name ,args :parameters ,parameters :forward ,forward :backward ,backward :hide-from-tree T))

(defmacro defmodel (name args &key parameters forward backward hide-from-tree)
  #|
  Define an node.
  Args: hide-from-tree ... if true, this node is detached from autograd. (that is, when backward, use backward defined in itself)
  |#
  (labels ((assure-args (x)
	     (if (or (equal (symbol-name x) "forward")
		     (equal (symbol-name x) "backward")
		     (equal (symbol-name x) "hide-from-tree")
		     (equal (symbol-name x) "parameters")
		     (equal (symbol-name x) "self")) ; i am not sure if it is really enough
		 (error "the name ~a is not allowed to use" (symbol-name x))
		 x)))
    (unless forward
      (error "insufficient forms"))
    `(defmacro ,name (&rest init-args &aux (constructor-name (gensym)))
       `(progn
	  (defstruct (,(gensym (symbol-name ',name))
		      (:constructor ,constructor-name (,@',args &aux ,@',parameters))
		      (:print-function (lambda (m stream k)
					 (declare (ignore k))
					 (render-simple-model-structure stream m))))
	   (hide-from-tree ,',hide-from-tree)
	   (parameters ',',(map 'list (lambda (x) (car x)) parameters))
	   (forward ,',(let ((largs (car forward))
			     (lbody (cdr forward))
			     (self-heap (gensym)))
			 (dolist (i largs) (assure-args i))
			 `(lambda ,(concatenate 'list (list self-heap) largs)
			    ,(if (null parameters)
				 `(declare (ignore ,self-heap)))
			    (macrolet ((self (name)
					 `(slot-value ,',self-heap ',name)))
			      ,@lbody))))
	   (backward ,',(if backward
			    (let ((largs (car backward))
				  (lbody (cdr backward))
				  (self-heap (gensym)))
			      (dolist (i largs) (assure-args i))
			      `(lambda ,(concatenate 'list (list self-heap) largs)
				 ,(if (null parameters)
				      `(declare (ignore ,self-heap)))
				 (macrolet ((self (name)
					      `(slot-value ,',self-heap ',name)))
				   ,@lbody)))
			    nil))
	   ,@',(map 'list (lambda (x) (assure-args (car x))) parameters))
	  (,constructor-name ,@init-args)))))


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

