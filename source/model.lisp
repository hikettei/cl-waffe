
(in-package :cl-waffe)

(declaim (inline call))

(defparameter *no-grad* nil)

(defgeneric call-forward  (self))
(defgeneric call-backward (self))

; TODO: Rename -> with-predict-mode
(defmacro with-no-grad (&body body)
  `(let ((no-grad-first *no-grad*))
     (setq *no-grad* t)
     ,@body
     (setq *no-grad* no-grad-first)
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
  (declare (optimize (speed 3) (safety 0) (space 0)))
  ; calculating op(x,y) -> result(x, y), state
  (let* ((result (apply
		  (the function (call-forward model)) args)))
    (declare (type (or null waffetensor list) result))
        
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
       :hide-from-tree nil
       :regard-as-node nil)))

(defmacro defnode (name args &key parameters forward backward optimize (regard-as-node t))
  `(defmodel ,name ,args :parameters ,parameters :forward ,forward :backward ,backward :hide-from-tree T :optimize ,optimize :regard-as-node ,regard-as-node))

(defun decide-thread-idx (&rest args)
  (let ((nm (find t args :test (lambda (_ x)
				 (declare (ignore _))
				 (waffetensor-thread-data x)))))
    (if nm
	(1+ (waffenodethread-thread-idx (waffetensor-thread-data nm)))
	0)))

(defun set-thread-data (th &rest args)
  (dolist (v args)
    (setf (waffetensor-thread-data v) th)))

(defmacro allocate-grad-id (slot-name tensor)
  `(let* ((thread (waffetensor-thread-data ,tensor)))
     (cl-waffe.backends.mgl:create-thread-idx thread
					      (gensym (symbol-name ',slot-name)))))

(defun free-caches (thread &optional (evacuate-num 0))
  (let ((caches-n (waffenodethread-cache-n thread)))
    (dotimes (i (- caches-n evacuate-num))
      (setf (waffenodethread-cache-n thread) i)
      (let ((cache-id (cl-waffe.backends.mgl:create-thread-idx thread)))
	(cl-waffe.caches:free-cache cache-id))))
  nil)

(defun cache-tensor (tensor thread)
  (typecase (data tensor)
    (mat 
     (cl-waffe.caches:with-cache
	 (tmp
	  tensor
	  :place
	  (let ((place nil))
	    (incf (waffenodethread-cache-n thread) 2)
	    (setq place (cl-waffe.backends.mgl:create-thread-idx thread))
	      place))
       (copy! (data tensor) tmp)
       tmp))
    (T (data tensor))))

(defmacro define-node-method (fname
			      name
			      args
			      body
			      hide-from-tree
			      optimize
			      &optional (is-node nil)
			      &aux (thread (gensym)))
  "The macro for defining node method. (:forward :backward in defmodel, defnode, defoptimizers)
  Also, the code for managing caches."
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
	   ; Utils that can be used in :forward and :backward
	   (macrolet ((self (name) `(slot-value ,',self-heap ',name))
		      (save-for-backward (name value)
			`(let ((smaller-value (detach ,value)))
			   (unless *no-grad*
			   (typecase (data smaller-value)
			     (mat
			      (cl-waffe.caches:with-cache
				  (tmp
				   smaller-value
				   :place ; todo
				   (allocate-grad-id ,name ,value))
				(!allow-destruct smaller-value)
				(copy! (data smaller-value) tmp)
				(setf (data smaller-value) tmp)			
				(setf (self ,name) smaller-value)))
			     (T (!allow-destruct smaller-value)
			      (setf (self ,name) smaller-value)))))))
	     ,(if is-node ; when the model is node method and step is ended, cl-waffe will clean caches
		  `(let ((,thread (thread (decide-thread-idx ,@args))))
		     (set-thread-data ,thread ,@args)
		     (let ((result (locally ,@body)))
		       ;(declare (type waffetensor &optional result))
		       (typecase result
			 (list (prog1
				   (map 'list
					(lambda (x)
					  (typecase (waffetensor-data x)
					    (mat
					     (setf (data x) (data x))))
					  x)
				    result)
				 (free-caches ,thread 0)))
			 (T
			  (prog1
			      (progn
				(typecase (waffetensor-data result)
				  (mat
				   (setf (data result) (data result))))
				result)
			    (free-caches ,thread 0))))))
		  `(locally ,@body))))
	 (defmethod ,fname ((self ,name))
	   (lambda (&rest node-inputs) (apply #',f-ident self node-inputs))))))

(defmacro defmodel (name
		    args
		    &key
		      parameters
		      (forward `((&rest args) (error ":forward isn't defined.")))
		      (backward `((&rest args) (error ":backward isn't defined.:"))) ; displaying name is todo
		      hide-from-tree
		      optimize
		      (regard-as-node nil))
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

    (if (and (not hide-from-tree) backward)
	(format t "Warning: backward with hide-from-tree=nil never called in backward processes~%"))
    
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
	 (define-node-method
	     call-forward
	     ,name
	     ,(car forward)
	     ,(cdr forward)
	     ,hide-from-tree
	     ,optimize
	     ,regard-as-node)
	 (define-node-method
	     call-backward
	     ,name
	     ,(car backward)
	     ,(cdr backward)
	     ,hide-from-tree
	     ,optimize
	     ,regard-as-node)))))

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


