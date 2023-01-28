
(in-package :cl-waffe)

(declaim (inline call))

(defparameter *no-grad* nil)

(defgeneric call-forward  (self))
(defgeneric call-backward (self))

; TODO: Rename -> with-predict-mode
(defmacro with-no-grad (&body body)
  `(progn
     (setq *no-grad* t)
     ,@body
     (setq *no-grad* nil)
     nil))

(defmacro with-calling-layers (input &rest layers)
  ; todo rewrite, cuz this definition is temporary
  `(let ((,input ,input))
       ,@(map 'list (lambda (layer)
		      (declare (type cons layer))
		      `(progn
			 (!allow-destruct ,input)
			 (setq ,input (call (self ,(car layer)) ,@(cdr layer)))))
	      layers)
     ,input))

(declaim (inline call))
(declaim (ftype (function (t &rest waffetensor) waffetensor) call))
(defun call (model &rest args)
  ; calculating op(x,y) -> result(x, y), state

  (if (slot-value model 'hide-from-tree) ;when node;
      (progn
	(map 'list #'(lambda (x)
		       (setf (waffetensor-is-node-tensor x) t))
	     args)))
  
  (let* ((result (apply (call-forward model) args)))
    (declare (type (or null waffetensor list) result))
    
    (etypecase result
      (list (dolist (v result)
	      (setf (waffetensor-is-node-tensor v) nil)))
      (waffetensor (setf (waffetensor-is-node-tensor result) nil)))
    
    (unless *no-grad*
      (if (slot-value model 'hide-from-tree) ;is model defined by defmodel?
	  (progn
	    (setf (waffetensor-backward result) t)
	    (setf (waffetensor-state result) model)
	    (setf (waffetensor-variables result) args)
	    (setf (waffetensor-is-ancestor-param result) (if (member-if #'(lambda (x)
				                                            (waffetensor-is-ancestor-param x))
								      args)
							   t
							   nil)))))
    result))

(defmacro with-model-list (&rest models)
  `(model-list ,models))

(defmacro is-waffe-model (model)
  `(and (slot-exists-p ,model 'parameters)
        (slot-exists-p ,model 'hide-from-tree)
        (slot-exists-p ,model 'forward)
        (slot-exists-p ,model 'backward)))

(defun find-variables (model)
  (let ((parameters `(T)))
    (labels ((search-param (m)
	       (cond
		 ((typep m 'cl-waffe:model-list)
		  (dolist (p (slot-value m
					 (car (slot-value m 'cl-waffe:parameters))))
		    (search-param p)))
		 ((is-waffe-model m)
		  (dolist (p (slot-value m 'parameters))
		    (search-param (slot-value m p))))
		 ((typep m 'WaffeTensor)
		  (push m parameters)))))
      (search-param model)
      (if (= (length parameters) 1)
	  (error "Could not find any parameter")
	  (butlast parameters)))))

(defmacro defoptimizer (name args &key parameters update optimize)
  `(progn
     (defmodel ,name ,args
       :parameters ,parameters
       :optimize ,optimize
       :forward ,update ;zero-grad
       :backward ((model) (dolist (p (find-variables model))
			    (setf (waffetensor-state p) nil)
			    (setf (waffetensor-backward p) nil)
			    (setf (waffetensor-variables p) nil)
			    (if (waffetensor-grad p) ; only params have grad
				(setf (waffetensor-grad p) `(nil nil)))
			    (setf (waffetensor-grad-tmp p) (make-grad-tmp)))
		nil)
     :hide-from-tree nil)))

(defmacro defnode (name args &key parameters forward backward optimize)
  `(defmodel ,name ,args :parameters ,parameters :forward ,forward :backward ,backward :hide-from-tree T :optimize ,optimize))

(defmacro define-node-method (fname name args body hide-from-tree optimize)
  (let ((f-ident   (gensym (symbol-name name)))
	(self-heap (gensym (symbol-name name))))
    `(progn
         (declaim (ftype (function (,name ,@(map 'list (lambda (x) (declare (ignore x)) `waffetensor) `,args)) (or null waffetensor)) ,f-ident))
	 (defun ,f-ident (,self-heap ,@args)
	   ,(if optimize
		`(declare (optimize (speed 3) (space 0) (safety 0))
			  (type ,name ,self-heap))
		`(declare (type ,name ,self-heap)))
	   ,(if hide-from-tree `(declare (type waffetensor ,@args)) nil)
	   (macrolet ((self (name) `(slot-value ,',self-heap ',name))
		      (save-for-backward (name value)
			`(let ((smaller-value (detach ,value)))
			   (!allow-destruct smaller-value)
			   (setf (waffetensor-is-node-tensor smaller-value)
				 t)
			   (setf (self ,name) smaller-value))))
	     ,@body))
	 (defmethod ,fname ((self ,name))
	   (lambda (&rest node-inputs) (apply #',f-ident self node-inputs))))))

(defmacro defmodel (name args &key parameters forward backward hide-from-tree optimize)
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

    (if (and hide-from-tree backward)
	(print "Warning: backward with hide-from-tree=nil never called in backward processes"))
    
    (prog1
      `(prog1
	   (defstruct (,name
		       (:print-function (lambda (m stream k)
					  (declare (ignore k))
					  (render-simple-model-structure stream m)))
		       (:constructor ,name (,@args &aux ,@(map 'list (lambda (x) `(,(car x) ,(second x))) parameters))))
	     (hide-from-tree ,hide-from-tree :type boolean)
	     (forward t :type boolean)
	     (backward ,(if backward t nil) :type boolean)
	     (parameters ',(map 'list (lambda (x) (assure-args (car x))) parameters))
	     ,@(map 'list (lambda (x) `(,(assure-args (car x)) ,(second x) ,@(cddr x))) parameters))
	   (define-node-method call-forward  ,name ,(car forward)  ,(cdr forward) ,hide-from-tree ,optimize)
	   (define-node-method call-backward ,name ,(car backward) ,(cdr backward) ,hide-from-tree ,optimize)))))

(defun render-simple-model-structure (stream model) ; Todo: More Details
  (format stream "[~a: ~a]" (if (slot-value model 'hide-from-tree)
				"Node "
				"Model")
	  (type-of model)))

(defun print-model (model)
  (fresh-line)
  (render-model-structure t model))

(defparameter *initial-indent-size* 4)

(defun render-model-structure (stream model &optional (indent-level 0) (total-param 0) (model-name "Model") (indent-increase 4) (prefix "Param"))
  ; Todo: More Details
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


