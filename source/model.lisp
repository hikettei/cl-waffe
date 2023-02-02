
(in-package :cl-waffe)

(declaim (inline call))

(defparameter *no-grad* nil
  "When t, some node will be ignored. see references below for details. default: nil")

(defparameter *in-node-method* nil)

(defgeneric call-forward  (self))
(defgeneric call-backward (self))

(defmacro with-no-grad (&body body &aux (no-grad-first (gensym)))
  "This macro is like with-predict-mode
   When you predicting your models, copying values for backward is waste.
   In this macro, *no-grad* become t, and won't make computation nodes.
   And in some operations, they will be faster."
  `(let ((,no-grad-first *no-grad*))
     (setq *no-grad* t)
     (prog1 (progn ,@body)
       (setq *no-grad* ,no-grad-first))))

(defmacro with-node-method-mode (&body body &aux (state-first (gensym)))
  `(let ((,state-first *in-node-method*))
     (setq *in-node-method* t)
     (prog1 (progn ,@body)
       (setq *in-node-method* ,state-first))))

(defmacro with-calling-layers (input &rest layers)
  "This macro allows to sequentially call layers.
   Example:
   (defmodel MLP (activation)
       :parameters ((layer1   (denselayer (* 28 28) 512 T activation))
   	            (layer2   (denselayer 512 256 T activation))
	            (layer3   (linearlayer 256 10 T)))
       :forward ((x)
	         (with-calling-layers x
	           (layer1 x)
 	           (layer2 x)
	           (layer3 x))))

   the argument input is a tensor.
   Refering each layers from (self) macro, destructively modifying x with the returned value.

  For the different arguments, this is ok.

  (with-calling-layers x
       (layer1 x 1 1)
       (layer2 1 x 2)
       (layer3 x y))

  Input: input, an tensor. layers an cons.
  Output: An tensor made by calling layers"
  
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
  "Calling Forward Step defined by defmodel, defnode, defoptimizer and recording each argument's variable slot (i.e. building computation node).
   When *no-grad* is t, nodes won't created.
   Input: model ... an structure object
          args ... tensors
   Output: An Tensor (where the tensor is defined by sysconst)"
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
  "Applying model-list.
   Input: models an list of models
   Output: [Model:Model-List]
   The model defined by model-list can be used like:
   (call (Model List Structure you defined) i args...) where i is the index for models"
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

(defmacro defoptimizer (name
			args
			&key
			  parameters
			  update
			  optimize
			  (document "An optimizer, defined by cl-waffe."))
  "Defining optimizers. Internally, This is paraphase of defmodel, which slot names are different.

Note: by calling :backward slot of this, that behaves as (zero-grad).


Input:

name ... The optimizer's structure and constructor will be defined based on name

args ... Initializer of the optimizer. **The first value of initializer is the hash-table that collected model's parameter where the key is fixnum from 0 to n. You have to save it.

&keys

parameters ... An parameters that it has. (the same as structure's slot)

Example: :parameters ((count 0)
                      (humans 0 :type fixnum)
                      (inited? nil))

update: when training and (update) is called, this slot is called and you optimizer your parameters.

optimize: when t, the :update slot is defined with (optimize (speed 3) (space 0) (debug 0)) Default: nil


document: docstring for optimizers

You can use your optimized by using deftrainer, see the references of that.

(The way you call optimizer without using deftrainer is coming soon...)

Example:

```lisp
(defoptimizer SGD (params &key (lr 1e-3))
  :optimize t
  :parameters ((params params :type hash-table)
               (lr lr :type single-float))
  :update (()
       (dotimes (i (hash-table-count (self params)))
         ; W(n+1) = W(n) - n * grad
         (!modify (gethash i (self params))) :+=
               (!mul (self lr) (grad (gethash i (self params))))```"
  `(progn
     (defmodel ,name ,args
       :parameters ,parameters
       :optimize ,optimize
       :forward ,update
       ;zero-grad
       :backward ((model) (dolist (p (find-variables model))
			    (setf (waffetensor-state p) nil)
			    (setf (waffetensor-backward p) nil)
			    (setf (waffetensor-variables p) nil)
			    (if (waffetensor-grad p) ; only params have grad
				(setf (waffetensor-grad p) `(nil nil)))
			    (setf (waffetensor-grad-tmp p) (make-grad-tmp)))
		nil)
       :hide-from-tree nil
       :document ,document
       :regard-as-node t)))

(defmacro defnode (name
		   args
		   &key
		     parameters
		     forward
		     backward
		     optimize
		     (regard-as-node t)
		     (document "An node, defined by cl-waffe."))
  "Defining computation nodes. Internally, This is just paraphase of defmodel, which slot names are different.

The argument is almost the same as defoptimizer, defmodel. So I will omit.

the slot :forward is called by (call Node-Object args...)

the slot :backward is called by (backward out) where out is scalar outputs.

When the slot :regard-as-node is nil, an optimizer in cl-waffe.caches regards this as model (i.e. argument could be destructed.) Default is t.

Note that :backward must return list, where that length corresponds with the length of input's argument, otherwise an error occurs when backward.

Note: cl-waffe do not modify parameters tensor in :parameters slot, so do not define this. If you want to use parameter, use defmodel instead.

In forward and backward, computation node isn't needed to be continuous.

But the last values must posses :thread-data. (do not think it deeply about it)

The node will be created with (name args), where name is the given name, args is the same.

And this can be called with call.

Example:

```lisp
(defnode AddTensor nil
  :optimize t
  :parameters nil
  :forward  ((x y)
	     (with-searching-calc-node :add x y))
  :backward ((dy) (list dy dy)))

(call (AddTensor) tensor1 tensor2)```"
  
  `(defmodel ,name ,args :parameters ,parameters :forward ,forward :backward ,backward :hide-from-tree T :optimize ,optimize :regard-as-node ,regard-as-node :document ,document))

(defun decide-thread-idx (&rest args)
  (let ((nm (find t args :test (lambda (_ x)
				 (declare (ignore _))
				 (waffetensor-thread-data x)))))
    (if nm
	(1+ (waffenodethread-thread-idx (waffetensor-thread-data nm)))
	0)))

(defun set-thread-data (&rest args)
  (dolist (v (cdr args))
    (setf (waffetensor-thread-data v) (car args))))

(defun free-caches (thread &optional (args-size 0) (evacuate-num 0))
  (let ((caches-n (waffenodethread-cache-n thread)))
    (loop for i fixnum upfrom args-size below (1+ (- caches-n evacuate-num))
	  do (progn (setf (waffenodethread-cache-n thread) i)
		    (let ((cache-id
			    (cl-waffe.backends.mgl:create-thread-idx thread)))
		      (cl-waffe.caches:free-cache cache-id)))))
  nil)

(defun enable-node-tensor (&rest args)
  (map 'list #'(lambda (x)
		 (prog1
		     (if (typep x 'waffetensor)
			 (waffetensor-path-through-node? x)
			 nil)
		   (when (and
			  (typep x 'waffetensor)
			  *in-node-method*)
		     (setf (waffetensor-path-through-node? x) t))))
       args))

(defun uncheck-node-tensor (first-states &rest args)
  (map 'list #'(lambda (x y)
		 (when (typep x 'waffetensor)
		   (setf (waffetensor-path-through-node? x) y))
		 nil)
       args first-states)
  nil)

(defmacro define-node-method (fname
			      name
			      args
			      body
			      hide-from-tree
			      optimize
			      &optional (is-node nil)
			      &aux (thread (gensym))
				   (is-top (gensym))
				   (state  (gensym)))
  "The macro for defining node method. (:forward :backward in defmodel, defnode, defoptimizers)
  Also, the code for managing caches."
  (let ((f-ident   (gensym (symbol-name name)))
	(self-heap (gensym (symbol-name name))))
    `(progn
         (declaim (ftype (function (,name ,@(map 'list (lambda (x) (declare (ignore x)) `waffetensor) `,args)) (or null waffetensor)) ,f-ident))
	 (defun ,f-ident (,self-heap ,@args)
	   ,(if optimize
		`(declare (optimize (speed 3) (space 0) (safety 1))
			  (type ,name ,self-heap))
		`(declare (type ,name ,self-heap)))
	   ,(if hide-from-tree `(declare (type waffetensor ,@args)) nil)
	   ; Utils that can be used in :forward and :backward
	   (macrolet ((self (name) `(slot-value ,',self-heap ',name))
		      (save-for-backward (name value)
			`(let ((thread-info (waffetensor-thread-data ,value))
			       (smaller-value (detach ,value)))
			   (unless *no-grad* ; is no-grad disabled?
			     (when (if *in-node-method*
				       (not
					(waffetensor-path-through-node? ,value))
				       t)
			       ; save-for-backward is ignored when 1. in with-no-grad macro. 2. Nodes connected like (Node) -> (Node) ; (in nodes, :forward :backward doesn't create grads.)
			       (cond
				 ((and (typep (data ,value) 'mat)
				       (not (null thread-info)))
				  (cl-waffe.caches:with-cache
				      (tmp
				       smaller-value
				       :place
				       (cl-waffe.backends.mgl:create-thread-idx
					thread-info)
				       :copy t)
				    (incf (waffenodethread-cache-n thread-info) 1)
				    (setf (self ,name) (const tmp))))
				 (T (!allow-destruct smaller-value)
				    (setf (self ,name) smaller-value))))))))
	     ,(if is-node
		  ; when method is for models, copy tensors, and caches.
		  `(let* ((,thread (thread (decide-thread-idx ,@args)))
			  (,is-top (= (waffenodethread-thread-idx ,thread) 0)))
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
				 (free-caches ,thread)
				 (when ,is-top
				   (set-thread-data nil ,@args))))
			 (T
			  (prog1
			      (progn
				(typecase (waffetensor-data result)
				  (mat
				   (setf (data result) (data result))))
				result)
			    (free-caches ,thread)
			    (when ,is-top
			      (set-thread-data nil ,@args)))))))
		  ; when method is for nodes, or optimizers
		  
		  `(let ((,state (enable-node-tensor ,@args)))
		     (declare (type list ,state))
		     (with-node-method-mode
		       (let ((result (progn ,@body))
			     (result-next-state (find t ,state)))
			 (uncheck-node-tensor ,state ,@args)
			 (typecase result
			   (list
			    (map 'list #'(lambda (x)
					   (setf (waffetensor-path-through-node? x) result-next-state)
					   x)
				 result))
			   (T
			    (setf (waffetensor-path-through-node? result) result-next-state)
			    result))))))))
	 (defmethod ,fname ((self ,name))
	   (lambda (&rest node-inputs) (apply #',f-ident self node-inputs))))))

(defmacro defmodel (name
		    args
		    &key
		      parameters
		      (forward `((&rest args) (error ":forward isn't defined.")))
		      (backward `((&rest args) (error ":backward isn't defined.:"))) ; Todo: displaying model name
		      hide-from-tree
		      optimize
		      (regard-as-node nil)
		      (document "An model, defined by cl-waffe"))
  "Defining model.
The arguments are the same as defnode, defoptimizer. So I will omit.

:backward will never called because cl-waffe follows computation node created in :forward step when backward.

However, when hide-from-tree is t, cl-waffe calls :backward.

The parameter tensor defined in :parameters will be updated by calling (update) and (backward out) in deftrainer's :step slot.

If you just would like to use cl-waffe's default features, you would mostly use this.

See the examples in the cl-waffe's repository.

In the :forward :backward slot, an macro `(self name)` defined by macrolet is supplied, this macro allows you access parameters in the model of current state. (Parameters aren't static.)

The model you defined is printable."
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
	     ,document
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
	     ,(not regard-as-node))
	 (define-node-method
	     call-backward
	     ,name
	     ,(car backward)
	     ,(cdr backward)
	     ,hide-from-tree
	     ,optimize
	     ,(not regard-as-node))))))

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


