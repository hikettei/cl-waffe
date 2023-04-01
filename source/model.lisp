
(in-package :cl-waffe)

#|
Here's
Utils for defnode/defmodel/defoptimizer
|#
(defparameter *in-node-method* nil)
(defparameter *model-arg-max-displaying-size* 20 "")

(defparameter *restart-non-exist-backend* t
  "When t, in the case when the specified backend doesn't exist, cl-waffe calls a standard implementation backend")

(defparameter *initial-form-forward*
  `((error "forward is nil")))

(defparameter *initial-form-backward*
  `((error "backward is nil")))

(defparameter *call-forward-features* (make-hash-table)
  "An hash-table which records all forward nodes")
(defparameter *call-backward-features* (make-hash-table)
  "An hash-table which records all backward nodes")

(defun register-features (features-table
			 node-name
			 fname
			 backend-name)
  (declare (optimize (speed 3))
	   (type hash-table features-table))
  (let ((features (or (gethash node-name features-table)
		      (make-hash-table))))
    (setf (gethash backend-name features) fname)
    (setf (gethash node-name features-table) features)
    nil))

(defun register-forward-features (node-name fname backend-name)
  (register-features *call-forward-features* node-name fname backend-name))

(defun register-backward-features (node-name fname backend-name)
  (register-features *call-backward-features* node-name fname backend-name))

(define-method-combination backend-dispatcher ()
  ((mgl-node-method  (backend-dispatcher . *))
   (external-methods (:external-node . *)))
  (:arguments node)
  (let ((extensions-call-form (mapcar
			       #'(lambda (method)
				   `(the (or function null)
					 (call-method ,method)))
			       external-methods)))
    `(or ,@extensions-call-form
	 ,(cond
	    ((null external-methods)
	     `(the function (call-method ,(first mgl-node-method))))
	    ((eql *default-backend* :mgl)
	     `(call-method ,(first mgl-node-method)))
	    (T
	     `(if (or
		   *restart-non-exist-backend*
		   (eql *default-backend* :mgl))
		  (call-method ,(first mgl-node-method))
		  (restart-case
		      (error (make-condition
			      'Backend-Doesnt-Exists
			      :kernel *default-backend*
			      :node ,node))
		    (restart-with-mgl-kernel ()
		      (call-method ,(first mgl-node-method)))))))
	 (progn
	   (error "cl-waffe: restarting was failed. ~a" ,node)))))

(defgeneric call-forward  (self) (:method-combination backend-dispatcher))
(defgeneric call-backward (self) (:method-combination backend-dispatcher))

(defmacro with-no-grad (&body body)
  "This macro is used in order to implict that codes below is ignored:
save-for-backward, creating new node object, using backward and processes for it.

For tasks in which grads are not required, using it helps better performance.

@begin[lang=lisp](code)
(with-no-grad
  (call (model) x))
@end[lang=lisp](code)"
  `(let ((*no-grad* t))
     ,@body))

(defmacro with-node-method-mode (&body body)
  `(let ((*in-node-method* t))
     ,@body))

(defmacro with-calling-layers (input &rest layers)
  "This macro allows to sequentially call layers.

the argument @cl:param(input) must be a tensor.

Refering each layers from (self) macro, destructively modifying x with the returned value.

Note: This macro supposes models to be returned a single tensor, not a list.

@begin[lang=lisp](code)

(defmodel MLP (activation)
   :parameters ((layer1   (denselayer (* 28 28) 512 T activation))
   	        (layer2   (denselayer 512 256 T activation))
	        (layer3   (linearlayer 256 10 T)))
   :forward ((x)
	     (with-calling-layers x
	       (layer1 x)
 	       (layer2 x)
               (layer3 x))))
@end[lang=lisp](code)

For the different arguments.

@begin[lang=lisp](code)
(with-calling-layers x
     (layer1 x 1 1)
     (layer2 1 x 2)
     (layer3 x y))
@end[lang=lisp](code)

Output: An last value of layers."  
  `(let ((,input ,input))
       ,@(map 'list (lambda (layer)
		      (declare (type cons layer))
		      `(progn
			 (!allow-destruct ,input)
			 (setq ,input (call (self ,(car layer)) ,@(cdr layer)))))
	      layers)
     ,input))

(declaim (ftype (function (t &rest waffetensor) (or null list waffetensor)) call))
(defun call (model &rest args)
  "Calls the forward steps which defined in: defnode, defmodel, defoptimizer.

All forward steps must be called through this function, otherwise the returned tensor doesn't have: computation nodes, thread-datum which supports performance.

Building computation nodes is ignored when *no-grad* is t.

@begin(deflist)
@term(model)
@def(Your initialized model/node/optimizer objects)
@term(args)
@def(Arguments :forward needs)
@end(deflist)

Example:
@begin[lang=lisp](code)
(defnode Add nil
  :optimize t
  :parameters nil
  :forward  ((x y)
	     (sysconst (+ (data x) (data y))))
  :backward ((dy) (list dy dy)))

(call (Add) (const 1.0) (const 1.0))
;=>Const(2.0)

@end[lang=lisp](code)

Output: Waffetensor of list which comprised of waffetensor."
  (declare (optimize (speed 3) (safety 0))
	   (notinline call))
  ; calculating op(x,y) -> result(x, y), state

  (when (model-list-p model)
    (return-from call
      (apply
       #'call
       (nth (the fixnum (data (car args)))
	    (model-list-mlist model))
       (cdr args))))
  
  (let* ((result (apply (the function (call-forward model)) args)))
    (declare (type (or null waffetensor list) result))

    (unless *no-grad*
      (if (slot-value model 'hide-from-tree) ;is model defined by defmodel?
	  (typecase result
	    (waffetensor
	     (setf (waffetensor-backward result) t)
	     (setf (waffetensor-state result) model)
	     (setf (waffetensor-variables result) args)
	     (setf (waffetensor-is-ancestor-param result)
		   (if (member-if #'waffetensor-is-ancestor-param args)
		       t
		       nil)))
	    (list
	     (mapc
	      #'(lambda (r)
		  (declare (type waffetensor r))
		  (setf (waffetensor-backward r) t)
		  (setf (waffetensor-state r) model)
		  (setf (waffetensor-variables r) args)
		  (setf (waffetensor-is-ancestor-param r)
		   (if (member-if #'waffetensor-is-ancestor-param args)
		       t
		       nil)))
	      result))
	    ;(t
	     ;(error "cl-waffe.defnode: Nodes must return a single tensor or list which consisted of waffetensor otherwise cl-waffe can't build up computation nodes..."))
	    )))
    result))

(defun update-computation-node ()
  )

(defun model-inlineable-p (model)
  (and (typep model 'list)
       (let ((name (car model)))
	 (and
	  (not (eql name 'model-list))
	  (gethash name *call-forward-features*)))))

;mlistはパスすればおk。callを再起する必要なし
;call modelは呼ばれるたびに、modelが同じStructureを受け取る必要がある
;+inline
(defmacro call1 (model
		 &rest inputs
		 ;&environment env
		 &aux
		   (model-type (type-of model))
		   (call-ident (gensym "CALL"))
		   (features (model-inlineable-p model)))
  #|
    If model is determined at compile-time. (e.g.: (call (ScalarAdd) (const 1.0) (const 1.0))), they inlined.
  |#
  (declare (optimize (speed 3)))
  (if features
      #|When there's a defined node|#
      (let ((backends)
	    (fnames))
	(maphash #'(lambda (backend-type function-symbol)
		     (push backend-type backends)
		     (push function-symbol fnames))
		 features)
	(if (= (the fixnum (length (the list backends))) 1)
	    `(locally
		 (declare (optimize (speed 3) (safety 1))
			  (inline ,(car fnames)))
	       (,(car fnames) ,model ,@inputs))
	    nil))
      (print :A)))
		 

(defun tmp-inf (model-name)
  (gethash (type-of model-name) *call-forward-features*))

(defnode ScalarAdd ()
  :forward-declaim (declaim (ftype (function (ScalarAdd waffetensor waffetensor) waffetensor) :forward))
  :forward ((x y)
	    (let ((x (data x))
		  (y (data y)))
	      (declare (type single-float x y))
	      (const (+ x y))))
  :backward ((dy) (list dy dy)))

(defnode ScalarSub ()
  :forward-declaim (declaim (ftype (function (ScalarSub waffetensor waffetensor) waffetensor) :forward))
  :forward ((x y)
	    (let ((x (data x))
		  (y (data y)))
	      (declare (type single-float x y))
	      (const (- x y))))
  :backward ((dy) (list dy dy)))

(define-node-extension cl-waffe::ScalarAdd
  :backend :numcl
  :forward ((x y) x)
  :backward ((dy) (list dy dy)))

(defun bench1 (&aux (node (ScalarAdd)))
  (declare (optimize (speed 3) (safety 0))
	   (inline |call-scalaradd-forward-mgl|))
  (time (dotimes (i 10000)
	  (|call-scalaradd-forward-mgl| node (const 1.0) (const 1.0)))))


(defun bench2 (&aux (node (ScalarAdd)))
  (declare (optimize (speed 3) (safety 0)))
  (with-no-grad
  (time (dotimes (i 10000)
	  (call node (const 1.0) (const 1.0))))))



(defmacro with-model-list (&rest models)
  "Applying model-list.

Input: models an list of models

Output: [Model:Model-List]

The model defined by model-list can be used like:

@c((call (Model-List) i args...)) where i is the index for models"
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
			initializer-arguments
			&key
			  parameters
			  (disassemble-update nil)
			  update-declaim
			  update
			  optimize
			  (document "An optimizer, defined by cl-waffe."))
  "Defining optimizers. Internally, This is paraphase of defmodel, which slot names are just different.

Note: by calling :backward slot, optimizers work as (zero-grad).

@begin(deflist)
@term(Name)
@def(The optimizer's structure and constructor will be defined based on name)

@term(Args)
@def(Initializer of the optimizer. The first value of initializer is the hash-table that collected model's parameter where the key is fixnum from 0 to n. You have to store it.)

@term(parameters)
@def(An parameters that it has.)

@term(update)
@def(when training and (update) is called, this slot is called and you optimizer your parameters.)

@term(optimize)
@def(when t, the :update slot is defined with (optimize (speed 3) (space 0) (debug 0)) Default: nil)

@term(document)
@def(docstring for optimizers. You can use string or (with-usage) macro)

@end(deflist)

Example:

@begin[lang=lisp](code)

;defoptimizer's args must start with params (symbol-name doesn't matter) which receives hash-table whose key is 1..n

(defoptimizer SGD (params &key (lr 1e-3))
  :optimize t
  :parameters ((params params :type hash-table)
               (lr lr :type single-float))
  :update (()
       (dotimes (i (hash-table-count (self params)))
         ; W(n+1) = W(n) - n * grad
         (!modify (gethash i (self params))) :+=
               (!mul (self lr) (grad (gethash i (self params)))))))

@end[lang=lisp](code)
"

  `(defobject ,name ,initializer-arguments
     :parameters ,parameters
     :optimize ,optimize
     :disassemble-forward ,disassemble-update
     :forward-declaim ,update-declaim
     :forward ,update
					;zero-grad
     :backward ((model) (dolist (p (find-variables model)) ; Todo Rewrite.
			  (setf (waffetensor-state p) nil)
			  (setf (waffetensor-backward p) nil)
			  (setf (waffetensor-variables p) nil)
			  (if (waffetensor-grad p) ; only params have grad
			      (setf (waffetensor-grad p) `(nil nil)))
			  (setf (waffetensor-grad-tmp p) (make-grad-tmp)))
		nil)
     :hide-from-tree nil
     :document ,document
     :regard-as-node t
     :object-type :optimizer))

(defmacro defnode (name
		   initializer-arguments
		   &key
		     parameters
		     (disassemble-forward nil)
		     forward-declaim
		     forward
		     (disassemble-backward nil)
		     backward-declaim
		     backward
		     (optimize nil)
		     (regard-as-node t)
		     (document "An node, defined by cl-waffe."))
  "Defining computation nodes.

defnode is useful when you want to define the derivative yourself.

@b(Note that parameter tensors in :parameter won't updated by optimizers.)

If you want to update params, define additional models.

@begin(deflist)

@term(regard-as-node)
@begin(def)
When the slot :regard-as-node is nil, an optimizer in cl-waffe.caches regards this as model (i.e. argument could be destructed.) Default is t.
@end(def)
@end(deflist)

Note that:

@begin(enum)
@item(:backward must return list, where that length corresponds with the length of input's argument, otherwise an error occurs when backward.)

@item(In forward and backward, computation node isn't needed to be continuous.However, the last values of :forward and :backward step, must posses :thread-data, which can be obtained by (waffetensor-thread-data tensor))
@end(enum)

Example:

@begin[lang=lisp](code)
(defnode AddTensor nil
  :optimize t
  :parameters nil
  :forward  ((x y)
	     (with-searching-calc-node :add x y))
  :backward ((dy) (list dy dy)))

(call (AddTensor) tensor1 tensor2)
@end[lang=lisp](code)"

  (if (null backward)
      (warn "The backward slot of ~a is undefined, which returns nil without cl-waffe being noticed." (symbol-name name)))
  
  `(defobject ,name ,initializer-arguments
     :parameters ,parameters
     
     :disassemble-forward ,disassemble-forward
     :forward-declaim ,forward-declaim
     :forward ,forward
     
     :disassemble-backward ,disassemble-backward
     :backward-declaim ,backward-declaim
     :backward ,backward
     
     :hide-from-tree T
     :optimize ,optimize
     :regard-as-node ,regard-as-node
     :document ,document
     
     :object-type :node))

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
		      (cl-waffe.caches:free-cache thread cache-id)))))
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
  (declare (optimize (speed 3))
	   (type list first-states args))
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
			      object-type
			      required-backend-symbol
			      &optional
				(is-node nil)
				(backend-name (list 'backend-dispatcher))
			      &aux (thread (gensym))
				   (is-top (gensym))
				   (state  (gensym)))
  "The macro for defining node method. (:forward :backward in defmodel, defnode, defoptimizers)
  Also, the code for managing caches."
  (declare (ignore hide-from-tree object-type))
  (let ((f-ident   (gensym (symbol-name name)))
	(self-heap (gensym (symbol-name name)))
	(vars (map 'list #'(lambda (x)
			     (typecase x
			       (list (car x))
			       (T x)))
		   (remove-if #'(lambda (x) (find x `(&optional &key &aux &rest)))
			      args))))
    `(progn
       #|(declaim (ftype
		 (function
		  (,name
		   ,@(map 'list (lambda (x)
				  (cond
				    ((find x `(&optional &key &aux &rest))
				     x)
				    (T `waffetensor)))
			  `,args))
		  (or null list waffetensor))
		 ,f-ident))|#
	 (defun ,f-ident (,self-heap ,@args)
	   ,(if optimize
		`(declare (optimize (speed 3) (space 0) (safety 1))
			  (type ,name ,self-heap))
		`(declare (type ,name ,self-heap))) ; This is needed to inline call-forwrd/backward.
;	   ,(if hide-from-tree `(declare (type waffetensor ,@vars)) nil)
	   ; Utils that can be used in :forward and :backward

	   ; Optimizer is required to use model in arguments
	   ;(when (not (eql ,object-type :optimizer))
	   ;    ,@(map 'list #'(lambda (variable)
		;		`(setq ,variable (typecase ,variable
		;				   (waffetensor ,variable)
		;				   (T (const ,variable)))))
		 ;     `,vars))
	   
	   (macrolet ((self (name) `(slot-value ,',self-heap ',name))
		      (model () `,',self-heap)
		      (save-for-backward (name value)
			`(let ((thread-info (waffetensor-thread-data ,value))
			       (smaller-value (detach ,value)))
			   (unless *no-grad* ; is no-grad disabled?
			     (when (if *in-node-method*
				       (not
					(waffetensor-path-through-node? ,value))
				       t)

					; save-for-backward is ignored when 1. in with-no-grad macro. 2. Nodes connected like (Node) -> (Node) ; (in nodes, :forward :backward doesn't create grads.)

			       (when (member t (list ,@',vars)
					     :test
					     #'(lambda (x y)
						 (eql x (waffetensor-is-ancestor-param y))))
				 (cond
				   ((and (typep (data ,value) 'mat)
					 (not (null thread-info)))
				    (cl-waffe.caches:with-cache
					(tmp
					 smaller-value
					 :place
					 (cl-waffe.backends.mgl:create-thread-idx thread-info)
					 :copy t)
				      (setf (self ,name) (const tmp))))
				   (T ;(!allow-destruct smaller-value)
				      (setf (self ,name) smaller-value)))))))))
	     (self hide-from-tree) ; avoid unused argument (model itself)
	     ,(if is-node
		  ; when method is for models.
		  ; WaffeNodeThread is created when called with model.
		  `(let* ((,thread (thread ; initialize thread-data with searching the toplevel ident of models.
				    (decide-thread-idx ,@vars)
				    (or (let ((k (find t (list ,@vars) :test #'(lambda (x y)
									   (declare (ignore x))
									   (waffetensor-thread-data y)))))
					  (if (and
					       k
					       (waffenodethread-belong-to (waffetensor-thread-data k)))
					      (waffenodethread-belong-to (waffetensor-thread-data k))
					      nil))
					(self model-ident))))
			  (,is-top (= (waffenodethread-thread-idx ,thread) 0)))
		     (set-thread-data ,thread ,@vars)
		     (let ((result (progn ,@body)))
		       ; no matter what returns model/node, cl-waffe won't error.
		       ;(declare (type waffetensor &optional result))
		       (typecase result
			 (list (prog1
				   result
				 (free-caches ,thread)
				 (when ,is-top
				   (set-thread-data nil ,@vars))))
			 (waffetensor
			  (prog1
			      (progn
				result)
			    (free-caches ,thread)
			    (when ,is-top
			      (set-thread-data nil ,@vars))))
			 (T result))))
		  ; when method is for nodes/optimizers
		  `(let ((,state (enable-node-tensor ,@vars)))
		     (declare (type list ,state))
		     (with-node-method-mode
		       (let ((result (progn ,@body))
			     (result-next-state (find t ,state)))
			 (uncheck-node-tensor ,state ,@vars)
			 (typecase result
			   (list
			    (map 'list #'(lambda (x)
					   (setf (waffetensor-path-through-node? x) result-next-state)
					   x)
				 result))
			   (T
			    (setf (waffetensor-path-through-node? result) result-next-state)
			    result))))))))
	 ; ,@backend-name -> backend-dispatcher / :external-node :numcl ...
	 (defmethod ,fname ,@backend-name ((self ,name))
	   (declare (optimize (speed 3) (safety 0))
		    (type ,name self))
	   (when (or
		  (eql :mgl ,required-backend-symbol)
		  (eql *default-backend* ,required-backend-symbol))
	     #'(lambda (&rest node-inputs)
		 (declare (optimize (speed 3) (safety 0))
			  (inline ,f-ident))
		 (apply #',f-ident self node-inputs)))))))

(defmacro defmodel (name
		    initializer-arguments
		    &key
		      (parameters nil)
		      (disassemble-forward nil)
		      forward-declaim
		      forward
		      (optimize nil)
		      (document "An model, defined by cl-waffe"))
  "This macro defines a cl-waffe model as @cl:param(name).

At the same time, a constructor @cl:param(name) is defined and you can initialize your model like:

@begin[lang=lisp](code)
(cl-waffe.nn:LinearLayer 100 20) ; => [Model: Linearlayer]
@end(code)

@title(Args)
@begin(deflist)

@term(name)
@def(Your model and constructor name)

@term(args)
@def(The arguments of a constructor)

@term(parameters)
@begin(def)

The parameters your model has.

Every time you initialize the model, the parameters are initialized.

Note that @cl:param(defmodel) behaves like class.

The arguments are the same as @cl:spec(defstruct)

Format Example: ((param-name param-initial-value &key (type your-type)))

@end(def)

@term(optimize)
@def(when t, your forward slot is defined with (declare (optimize (speed 3) (space 0) (debug 0))). It helps faster training after you ensured debugged.)

@term(forward)
@begin(def)

Define here the forward propagation of your model.

When backward, @b(Automatic differentiation applies).

@end(def)

@end(deflist)
"
  `(defobject ,name ,initializer-arguments
     :parameters ,parameters
     :disassemble-forward ,disassemble-forward
     :forward-declaim ,forward-declaim
     :forward ,forward
     :optimize ,optimize
     :object-type :model
     :document ,document))

(defmacro define-node-extension (name
				 &key
				   optimize
				   backend
				   (disassemble-forward nil)
				   forward-declaim
				   forward
				   (disassemble-backward nil)
				   backward-declaim
				   backward)
  "Adds a new backend to the defined node.

The type of backend is managed by keywords. The backend defined in defnode is always :mgl.

Defined backends can be switched by the macro @c((with-backend backend)).

As long as *restart-non-exist-backend* is t, when a computation node reaches a backend that is not defined, :mgl is called, otherwise the condition backend-doesnt-exists will occurs.

Example:

@begin[lang=lisp](code)
(define-node-extension cl-waffe::AddTensor
  :backend :test-backend
  :forward ((x y)
        (const (+ 1 1)))
  :backward ((dy)
         (list dy dy)))

(with-backend :mgl
   (print (!add 10 10))) ;=> Const(20)

(with-backend :test-backend
   (print (!add 10 10))) ;=> Const(2)

(with-backend :hogehoge
   (print (!add 10 10))) ; => Const(20)

(let ((*restart-non-exist-backend* nil))
    (with-backend :hogehoge
        (print (!add 10 10)))) ;=> Evaluation aborted on #<CL-WAFFE::BACKEND-DOESNT-EXISTS {100FA18C43}>.
@end[lang=lisp](code)
"
  `(progn
     (assert (not (equal (symbol-name ',name)
			 "MODEL-LIST"))
	     nil
	     "define-node-extension is failed because cl-waffe::model-list is attempted to overwrite. This node is prohibited to extend")
     (define-node-function
	 :forward
       ,name
       ,forward-declaim
       ,(car forward)
       ,(or (cdr forward)
	    *initial-form-forward*)
       :node
       ,backend
       :disassemble-me ,disassemble-forward)

     (define-node-function
	 :backward
       ,name
       ,backward-declaim
       ,(car backward)
       ,(or (cdr backward)
	    *initial-form-backward*)
       :node
       ,backend
       :disassemble-me ,disassemble-backward)
     
     (define-node-method
	 call-forward
	 ,name
	 ,(car forward)
	 ,(cdr forward)
	 nil
	 ,optimize
	 :node
	 ,backend
	 nil
	 (:external-node ,backend))
     (define-node-method
	 call-backward
	 ,name
	 ,(car backward)
	 ,(cdr backward)
	 nil
	 ,optimize
	 :node
	 ,backend
	 nil
	 (:external-node ,backend))
     nil))

(defun generate-function-name (function-type
			       structure-name
			       backend-type)
					; todo: export it for debugging.
  "(generate-function-name :forward 'areftensor :mgl)
   ;=> |gen-by-waffe4682call-AREFTENSOR-FORWARD-MGL|"
  (declare (optimize (speed 3))
           (type keyword function-type backend-type)
	   (type symbol structure-name))
  (assert (find function-type `(:forward :backward))
	  nil
	  "in cl-waffe's internal, Assertion Failed with function-type = [:forward :backward]")
  (intern
   (string-downcase
    (concatenate 'string
		 "CALL-"
		 (symbol-name structure-name)
		 "-"
		 (symbol-name function-type)
		 "-"
		 (symbol-name backend-type)))))

(defun replace-declaim-forms-with-fname (forward-or-backward
					 function-name
					 forms)
  (map-tree
   #'(lambda (code)
       (typecase code
	 (keyword
	  (if (eql code forward-or-backward)
	      function-name
	      code))
	 (T
	  code)))
   forms))

(defmacro with-object-macrolet-forms (self-heap vars &body body)
  "vars - the list of symbols"
  `(macrolet ((self (name)
		`(slot-value ,,self-heap ',name))
	      (model () ,self-heap)
	      (save-for-backward (name value)
		`(let ((thread-info (waffetensor-thread-data ,value))
		       (smaller-value (detach ,value)))
		   (unless *no-grad* ; is no-grad disabled?
		     (when (if *in-node-method*
			       (not
				(waffetensor-path-through-node? ,value))
			       t)
		       (when (member t (list ,@',vars)
				     :test
				     #'(lambda (x y)
					 (eql x (waffetensor-is-ancestor-param y))))
			 (cond
			   ((and (typep (data ,value) 'mat)
				 (not (null thread-info)))
				    (cl-waffe.caches:with-cache
					(tmp
					 smaller-value
					 :place
					 (cl-waffe.backends.mgl:create-thread-idx thread-info)
					 :copy t)
				      (setf (self ,name) (const tmp))))
			   (T ;(!allow-destruct smaller-value)
			    (setf (self ,name) smaller-value)))))))))
     (progn
       (model)
       nil
       ,@body)))

(defmacro with-define-function-in-defmodel-way
    (vars
     &body
       body
     &aux
       (thread (gensym "thread"))
       (is-top (gensym "is-top")))
  `(let* ((,thread (thread
		    (decide-thread-idx ,@vars)
		    (or (let ((k (find t (list ,@vars)
				       :test #'(lambda (x y)
						 (declare (ignore x))
						 (waffetensor-thread-data y)))))
			  (if (and
			       k
			       (waffenodethread-belong-to
				(waffetensor-thread-data k)))
			      (waffenodethread-belong-to
			       (waffetensor-thread-data k))
			      nil))
			(self model-ident))))
	  (,is-top (= (waffenodethread-thread-idx ,thread) 0)))
     (set-thread-data ,thread ,@vars)
     (let ((result (multiple-value-bind (result) (progn ,@body) ; FIXME: (values x y) is not available in defmodel
		     result)))
       (typecase result
	 (list (prog1
		   result
		 (free-caches ,thread)
		 (when ,is-top
		   (set-thread-data nil ,@vars))))
	 (waffetensor
	  (prog1
	      (progn
		result)
	    (free-caches ,thread)
	    (when ,is-top
	      (set-thread-data nil ,@vars))))
	 (T result)))))

(defmacro with-define-function-in-defnode-way
    (vars
     &body
       body
     &aux
       (state (gensym "STATE")))
  `(let ((,state (enable-node-tensor ,@vars)))
     (declare (type list ,state))
     (with-node-method-mode
       (let ((result
	       (multiple-value-bind (result) (progn ,@body)
		 result))
	     (result-next-state
	       (find t ,state)))
	 (uncheck-node-tensor ,state ,@vars)
	 (typecase result
	   (list
	    (map
	     'list
	     #'(lambda (x)
		 (setf
		  (waffetensor-path-through-node? x)
		  result-next-state)
		 x)
	     result))
	   (T
	    (setf
	     (waffetensor-path-through-node? result)
	     result-next-state)
	    result))))))

(defmacro define-node-function (forward-or-backward
				structure-name
				declaim-forms
				args
				body
				object-type ; :node :model etc
				backend-type
				&key
				  (disassemble-me t))
  "forward-or-backward :forward or :backward
   Strucutre name defined by defobject (e.g.: areftensor)"
  (let ((function-name (generate-function-name
			forward-or-backward
			structure-name
			backend-type))
	(self (gensym "self"))
	(tmp-fname (if disassemble-me
		       (gensym "nodedebug")))
	(args-symbols (reverse (get-params args))))
    (multiple-value-bind (body declarations docs)
	(alexandria:parse-body body :documentation t)
      (case forward-or-backward
	(:forward
	 (register-forward-features structure-name
				    function-name
				    backend-type))
	(:backward
	 (register-backward-features structure-name
				     function-name
				     backend-type))
	(T (error "")))
      `(progn
	 (declaim (inline ,function-name))
	 ,(replace-declaim-forms-with-fname
	   forward-or-backward
	   function-name
	   declaim-forms)
	 ,(if disassemble-me
	      (replace-declaim-forms-with-fname
	       forward-or-backward
	       tmp-fname
	       declaim-forms))
	 ,(if disassemble-me
	      `(defun ,tmp-fname (,self ,@args)
		 ,@declarations
		 ,docs
		 (with-object-macrolet-forms ',self ,args-symbols
		   ,@body)))
	 (defun ,function-name (,self ,@args)
	   ,docs
	   ,@declarations
	   (with-object-macrolet-forms ',self ,args-symbols
	     ,(if (find object-type `(:node :optimizer))
		  `(with-define-function-in-defnode-way ,args-symbols
		     ,@body)
		  `(with-define-function-in-defmodel-way ,args-symbols
		     ,@body))))
	 ,(if disassemble-me
	      `(disassemble #',tmp-fname))
	 nil))))

(defmacro defobject (name
		     args
		     &key
		       parameters
		       disassemble-forward
		       forward-declaim
		       forward
		       disassemble-backward
		       backward-declaim
		       backward
		       hide-from-tree
		       optimize
		       (regard-as-node nil)
		       (document "An object, defined by cl-waffe")
		       (object-type :object))
  "Defining cl-waffe's object
When hide-from-tree is t, autograds are ignored.
When regard-as-node is nil, the forward and backward is defined as the node.
the object-type indicates the type of document format."
  (declare (type boolean disassemble-forward disassemble-backward))
  (labels ((assure-args (x)
	     (declare (type symbol x))
	     (if (or (equal (symbol-name x) "FORWARD")
		     (equal (symbol-name x) "BACKWARD")
		     (equal (symbol-name x) "HIDE-FROM-TREE")
		     (equal (symbol-name x) "PARAMETERS")
		     (equal (symbol-name x) "MODEL-IDENT")
		     (equal (symbol-name x) "OBJECT-TYPE")
		     (equal (symbol-name x) "SELF"))
		 (invaild-slot-error (symbol-name x) object-type)
		 x)))

    (let* ((document (eval document))
	   (doc-output (typecase document
			(string document)
			(waffeobjectusage
			 (build-docstring document object-type))
			(T "None"))))

      (when (null forward)
	(warn "The forward slot of ~a is undefined, which returns nil without cl-waffe noticing."
	      (symbol-name name)))

      `(progn
	   (defstruct (,name
		       (:print-function (lambda (m stream k)
					  (declare (ignore k))
					  (render-simple-model-structure stream m)))
		       (:constructor
			   ,name
			   (,@args &aux (model-ident (gensym "W"))
				     ,@(map 'list (lambda (x) `(,(car x) ,(second x))) parameters))))
	     ,doc-output
	     (model-ident ,(gensym "W") :type symbol)
	     (hide-from-tree ,hide-from-tree :type boolean)
	     (forward t :type boolean)
	     (object-type ,object-type :type keyword)
	     (backward ,(if backward t nil) :type boolean)
	     (parameters ',(map 'list (lambda (x) (assure-args (car x))) parameters))
	     ,@(map 'list (lambda (x) `(,(assure-args (car x)) ,(second x) ,@(cddr x))) parameters))

	   (define-node-function
	       :forward
	     ,name
	     ,forward-declaim
	     ,(car forward)
	     ,(or (cdr forward)
		  *initial-form-forward*)
	     ,object-type
	     :mgl
	     :disassemble-me ,disassemble-forward)

	   (define-node-function
	       :backward
	     ,name
	     ,backward-declaim
	     ,(car backward)
	     ,(or (cdr backward)
		  *initial-form-backward*)
	     ,object-type
	     :mgl
	     :disassemble-me ,disassemble-backward)
	   
	   (define-node-method
	       call-forward
	       ,name
	       ,(car forward)
	       ,(cdr forward)
	       ,hide-from-tree
	       ,optimize
	       ,object-type
	       :mgl
	       ,(not regard-as-node))
	   (define-node-method
	       call-backward
	       ,name
	       ,(car backward)
	       ,(cdr backward)
	       ,hide-from-tree
	       ,optimize
	       ,object-type
	       :mgl
	       ,(not regard-as-node))
	   nil))))

(defun render-simple-model-structure (stream model) ; Todo: More Details
  (case (slot-value model 'cl-waffe::object-type)
    (:node
     (format stream "<Node: ~a{~a}>"
	     (type-of model)
	     (slot-value model 'cl-waffe::model-ident)))
    (:model
     (format stream "<Model: ~a{~a}("
	     (type-of model)
	     (slot-value model 'cl-waffe::model-ident))
     (when (not (= (length (slot-value model 'cl-waffe::parameters)) 0))
       (format stream "~%")
       (dolist (param (slot-value model 'cl-waffe::parameters))
	 (let ((val (slot-value model param)))
	   (cond
	     ((is-waffe-model val)
	      (format stream "    <Model: ~a -> ~a{~a} ...>~%"
		      param
		      (type-of val)
		      (slot-value val 'model-ident)))
	     (T
	      (format stream "    ~a : ~a~%"
		      param
		      (let ((seq (format nil "~a" val)))
			(concatenate 'string
				     (subseq seq 0 (min *model-arg-max-displaying-size* (length seq)))
				     (if (<= (length seq) *model-arg-max-displaying-size*)
					 ""
					 "...")))))))))
     (format stream ")>"))
    (:optimizer
     (format stream "<Optimizer: ~a{~a}~%"
	     (type-of model)
	     (slot-value model 'cl-waffe::model-ident))
     (let ((total-param 0))
       (dolist (param (slot-value model 'cl-waffe::parameters))
	 (let ((val (slot-value model param)))
	   (typecase val
	     (hash-table
	      (dotimes (i (hash-table-count val))
		(typecase (gethash i val)
		  (waffetensor
		   (incf total-param (!size (gethash i val))))
		  (mat
		   (incf total-param (mat-size (gethash i val))))
		  (T nil)))
	      (format stream "    Param: ~a~%" val))
	     (T
	      (format stream "    ~a : ~a~%"
		      param
		      (let ((seq (format nil "~a" val)))
			(concatenate 'string
				     (subseq seq 0 (min *model-arg-max-displaying-size* (length seq)))
				     (if (<= (length seq) *model-arg-max-displaying-size*)
					 ""
					 "..."))))))))
       (format stream "    [Total Param]: ~a~%>" total-param)))
    (T
     (format stream "[OBJECT: ~a {~a}]"
	     (type-of model)
	     (slot-value model 'cl-waffe::model-ident)))))

(defun print-model (model &optional (stream t))
  (format stream "~%")
  (let ((*total-param-size* 0))
    (render-model-structure stream model)
    (format stream "~% -(+) Total Param: ~a" *total-param-size*)
    nil))

(defun trim-string (str max-length)
  (concatenate 'string
	       (subseq str 0 (min (length str) max-length))
	       (if (< max-length (length str))
		   "..."
		   "")))

(defun trim-string1 (str max-length)
  (concatenate 'string
	       (subseq str 0 (min (length str) max-length))
	       (if (< max-length (length str))
		   ".."
		   "")))

(defun build-table-content (table-size value &key (end-with "|") (space "_"))
  (let* ((string-to-display (format nil "~a" value))
	 (size (length string-to-display))
	 (tsize (- table-size 1)))
    "min(table-size) = 5"
    (with-output-to-string (out)
      (cond
	((> size (- table-size 2))
	 ; needs to be trimmed.
	 (format out "~a"
		 (let* ((result
			  (trim-string1
			   string-to-display
			   (- table-size 4))))
		   result)))
			
	(T
	 (let ((displacement
		 (round (- (/ tsize 2) (/ (length string-to-display) 2)))))
	   (dotimes (i (1- displacement)) ; first=|
	     (format out space))
	   (format out string-to-display)
	   (dotimes (i (max 0
			    (- tsize
				   (+ (length string-to-display)
				      displacement))))
	     (format out space)))))
      (format out end-with)
      out)))

(defparameter *initial-indent-size* 4)
(defparameter *total-param-size* 0)

(defun render-model-structure (stream model
			       &optional
				 (indent-level 0)
				 (total-param 0)
				 (model-name "Model")
				 (indent-increase 4)
				 (prefix "Param")
				 (node-points nil))
  ; Todo: More Details

  (if (is-waffe-model model)
      (if (typep (slot-value model 'parameters) 'list)
	  (labels ((indent-with (code &optional (space NIL))
		     (dotimes (nth (+ indent-level (if space *initial-indent-size* 0)))
		       (if (find nth node-points)
			   (format stream code)
			   (format stream code)))))
	    (indent-with "–")
	    (format stream "––– <~a ~a{~a}>~%"
		    model-name
		    (type-of model)
		    (slot-value model 'cl-waffe::model-ident))
	    (let ((param-values nil)
		  (constants    nil)
		  (next-layers nil))
	      (dolist (param-name (slot-value model 'cl-waffe::parameters))
		(let ((value (slot-value model param-name)))
		  (cond
		    ((is-waffe-model value)
		     (setq next-layers `(,@next-layers
					 (,param-name . ,value))))
		    ((typep value 'waffetensor)
		     (setq param-values `(,@param-values
					  (,param-name . ,value))))
		    (T
		     (setq constants `(,@constants
				       (,param-name . ,value)))))))

	      (when constants
		(indent-with " " T)
		(format stream "|")
		(let ((i 0)
		      (spaces nil))
		  (dolist (c constants)
		    (when c
		      (unless (= i 0)
			(format stream "|"))
		      (incf i 1)
		      (push (length (format nil "––~a––" (car c))) spaces)
		      (format stream "-~a-" (car c))))
		  (format stream "|")
		  (setq i 0)
		  (setq spaces (reverse spaces))
		  (format stream "~%")
		  (indent-with " " T)
		  (format stream "|")
		  (dolist (c constants)
		    (when c
		      (format stream
			      (build-table-content
			       (pop spaces)
			       (cdr c))))))
		(format stream "~%"))
	      (when param-values
		(indent-with " " T)
		(let ((max-param-name-size
			(loop for i fixnum upfrom 0 below (length param-values)
			      maximize (length (format nil " ~a  " (car (nth i param-values))))))
		      (shape-str-size
			(loop for i fixnum upfrom 0 below (length param-values)
			      maximize (length (format nil " ~a  " (!shape (cdr (nth i param-values))))))))

		  (mapc #'(lambda (tensor)
			    (incf *total-param-size* (!size (cdr tensor))))
			param-values)
		  
		  (format stream "|–~a|"
			  (build-table-content
			   max-param-name-size
			   "slot"
			   :end-with ""
			   :space "–"))
		  (format stream "–~a|–trainable–|~%"
			  (build-table-content
			   shape-str-size
			   "shape"
			   :end-with ""
			   :space "–"))
		  (dolist (p param-values)
		    (indent-with " " T)
		    (format stream " ~a->"
			    (build-table-content
			     max-param-name-size
			     (format nil "~a" (car p))
			     :end-with ""
			     :space " "))

		    (format stream " ~a      ~a~%"
			    (build-table-content
			     shape-str-size
			     (format nil "~a" (!shape (cdr p)))
			     :end-with ""
			     :space " ")
			    (if (null (slot-value (cdr p) 'cl-waffe::grad))
				"X"
				"O")))))
	      (when next-layers
		(dolist (layer next-layers)
		  (cond
		    ((model-list-p (cdr layer))
		     (let ((models (slot-value (cdr layer) 'cl-waffe::mlist)))
		       (dotimes (i (length models))
			 (render-model-structure
			  stream
			  (nth i models)
			  (+ indent-level 2)
			  total-param
			  (format nil "~a's ~a[~ath] ="
				  (symbol-name (type-of model))
				  (symbol-name (car layer))
				  i)
			  indent-increase
			  ""
			  nil))))
		    (T
		     (render-model-structure
		      stream
		      (cdr layer)
		      (+ indent-level indent-increase)
		      total-param
		      (format nil "~a's ~a ="
			      (symbol-name (type-of model))
			      (symbol-name (car layer)))
		      indent-increase
		      ""
		      (if (<= (length next-layers) 1)
			  node-points
			  (progn
			    (pop node-points)
			    `(,@node-points ,(+ indent-increase indent-level))))))))))

	    (if (= indent-level 0)
		(progn
		  (format stream "")))))
	  ;in the end of model
      (labels ((indent-with ()
		 (dotimes (_ (+ indent-level *initial-indent-size*))
		   (format stream " "))))
	(if (typep model 'WaffeTensor)
	    (progn
	      (indent-with)
	      (format stream "(+)~a:~a~C" prefix (!shape model) #\newline))))))


