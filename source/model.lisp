
(in-package :cl-waffe)

#|
Here's
Utils for defnode/defmodel/defoptimizer
|#
(defparameter *in-node-method* nil)
(defparameter *model-arg-max-displaying-size* 20 "(print-model model) uses it. the argument content which longer than it, will be omitted.")

(defparameter *restart-non-exist-backend* t
  "When t, in the case when the specified backend doesn't exist, cl-waffe calls a standard implementation backend")

(defparameter *initial-form-forward*
  `((unimplemented-error "The :forward is undefined.")))

(defparameter *initial-form-backward*
  `((unimplemented-error "The :backward is undefined")))

(defparameter *ignore-inlining-info* nil
  "When t, cl-waffe fail to print indications on inlining.")

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

(defmacro with-no-grad (&body body)
  "Below this macro, the parameter *no-grad* become t, which means: some operations are forcibly ignored. (e.g.: save-for-backward, building computation nodes)
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


(defun call-inlined-forward (model &rest inputs)
  (redefine-call-inline-forward)
  (apply #'call-inlined-forward model inputs))

(defun call-inlined-backward (model &rest inputs)
  (redefine-call-inline-backward)
  (apply #'call-inlined-backward model inputs))

(defmacro call-backward (model &rest inputs)
  "calls the given model's backward, with inputs."
  `(call-inlined-backward ,model ,@inputs))

(defun build-backend-case (features
			   model
			   inputs
			   &aux (keys (hash-table-keys features)))
  (declare (type hash-table features))
  ; quite model and each inputs
  (if (= (length keys) 1)
      (progn
	`(apply
	  #',(gethash (car keys) features)
	  ,model
	  ,inputs))
      (progn
	`(case *default-backend*
	   ,@(loop for i fixnum upfrom 0 repeat (length keys)
		   collect `(,(nth i keys)
			     (apply
			      #',(gethash (nth i keys) features)
			      ,model
			      ,inputs)))
	   (T
	    ,(let ((default (car (last keys)))
		   (defaultfunc (gethash (car (last keys)) features)))
	       (assert (eql default :mgl)
		       nil
		       "cl-waffe:call Assertion Failed with default-backend != :mgl. Load cl-waffe's defnode first, and then load extensions.")
	       (if *restart-non-exist-backend*
		   `(apply
		     #',defaultfunc
		     ,model
		     ,inputs)
		   `(restart-case
			(error (make-condition
				'Backend-Doesnt-Exists
				:kernel *default-backend*
				:node ,model))
		      (restart-with-mgl-kernel ()
			(apply #',defaultfunc ,model ,inputs))))))))))

(defparameter *inlined-forward-retry-p* nil)

(defun redefine-call-inline-forward ()
  (let ((new-definition (generate-call-inline-forward)))
    (setf (symbol-function 'cl-waffe::call-inlined-forward)
	  new-definition)
    t))

(defun redefine-call-inline-backward ()
  (let ((new-definition (generate-call-inline-backward)))
    (setf (symbol-function 'cl-waffe::call-inlined-backward)
	  new-definition)
    t))

(defun generate-call-inline-forward ()
  (let ((keys (hash-table-keys *call-forward-features*))
	(functions))
     
    #|Todo: Add Event to avoid consume a lot of time to compile.|#

    (unless *ignore-inlining-info*
      (format t "~%[INFO] Inlining call-forward... Total Features: ~a~%To disable this, set cl-waffe:*ignore-inlining-info* t~%" (length keys)))

    (mapc
     #'(lambda (key)
	 (maphash #'(lambda (backend-name function-name)
		      (declare (ignore backend-name))
		      (push function-name functions))
		  (gethash key *call-forward-features*)))
     keys)
    
     (macrolet ((output-function ()
		  ``#'(lambda (model &rest inputs)
		       (declare (optimize (speed 3))
				(inline ,@functions))
		       (typecase model
			 ,@(loop for i fixnum upfrom 0 below (length keys)
				 collect `(,(nth i keys)
					   ,(build-backend-case
					     (gethash
					      (nth i keys)
					      *call-forward-features*)
					     'model
					     'inputs)))
			 (T
			  (if *inlined-forward-retry-p*
			      (nosuchnode-error "cl-waffe attempted to call forward of ~a but couldn't find such a node. ~%Please check:~%Is the node really defined? or the dependencies are loaded correctly?" model)
			      (let ((*inlined-forward-retry-p* t))
				(locally (declare (notinline call-inlined-forward))
				  (redefine-call-inline-forward)
				  (apply #'call-inlined-forward model inputs)))))))))
       ;(print (macroexpand (output-function)))
       (eval (output-function)))))

(defparameter *inlined-backward-retry-p* nil)

(defun generate-call-inline-backward ()
  (let ((keys (hash-table-keys *call-backward-features*))
	(functions))

    (unless *ignore-inlining-info*
      (format t "~%[INFO] Inlining call-backward... Total Features: ~a~%To disable this, set cl-waffe:*ignore-inlining-info* t~%" (length keys)))

    (mapc
     #'(lambda (key)
	 (maphash #'(lambda (backend-name function-name)
		      (declare (ignore backend-name))
		      (push function-name functions))
		  (gethash key *call-backward-features*)))
     keys)
    
     (macrolet ((output-function ()
		  ``#'(lambda (model &rest inputs)
		       (declare (optimize (speed 3))
				(inline ,@functions))
		       (typecase model
			 ,@(loop for i fixnum upfrom 0 below (length keys)
				 collect `(,(nth i keys)
					   ,(build-backend-case
					     (gethash
					      (nth i keys)
					      *call-backward-features*)
					     'model
					     'inputs)))
			 (T
			  (if *inlined-backward-retry-p*
			      (nosuchnode-error "cl-waffe attempted to call backward of ~a but couldn't find such a node. ~%Please check:~%Is the node really defined? or the dependencies are loaded correctly?" model) ; todo more conditions
			      (let ((*inlined-backward-retry-p* t))
				(locally (declare (notinline call-inlined-backward))
				  (redefine-call-inline-backward)
				  (apply #'call-inlined-backward model inputs)))))))))
       ;(print (macroexpand (output-function)))
       (eval (output-function)))))

(defun model-inlineable-p (model)
  (and (typep model 'list)
       (let ((name (car model)))
	 (and
	  (not (eql name 'model-list))
	  (gethash name *call-forward-features*)))))

(defmacro call (model
		&rest inputs
		&aux
		  (features (model-inlineable-p model)))
  "calls the given model's forward slot with inputs."
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
	    `(locally
		 (declare (optimize (speed 3) (safety 1))
			  (inline ,@fnames))
	       (case *default-backend*
		 ,@(loop for i fixnum upfrom 0 repeat (length backends)
			 collect `(,(nth i backends)
				   (,(nth i fnames) ,model ,@inputs)))
		 (T
		  ,(let ((default (car (last backends)))
			 (defaultfunc (car (last fnames))))
		     (assert (eql default :mgl)
			     nil
			     "cl-waffe:call Assertion Failed with default-backend != :mgl. Load cl-waffe's defnode first, and then load extensions.")
		     (if *restart-non-exist-backend*
			 `(,defaultfunc
			   ,model
			   ,@inputs)
			 `(restart-case
			      (error (make-condition
				      'Backend-Doesnt-Exists
				      :kernel *default-backend*
				      :node ,model))
			    (restart-with-mgl-kernel ()
			      (,defaultfunc ,model ,@inputs))))))))))
      (progn
	`(let* ((model ,model)
		(inputs (list ,@inputs)))
	   (if (typep model 'model-list)
	       (progn
		 (setq model (nth (data (car inputs))
				  (model-list-mlist model)))
		 (setq inputs (cdr inputs))
		 (assert (not (typep model 'model-list))
			 nil
			 "cl-waffe.call: Assertion failed because model-list can't posses model-list as a element.")))
	   (locally (declare (optimize (speed 3))
			     #+sbcl(sb-ext:maybe-inline call-inlined-forward)
			     #-sbcl(inline call-inlined-forward)
			     )
	     (apply #'call-inlined-forward model inputs))))))

(declaim (ftype (function (keyword t) function) get-nodefunction-caller))
(defun get-nodefunction-caller (forward-or-backward
				model
				&aux
				  (node-type (type-of model)))
  (declare (optimize (speed 3) (safety 0)))
  (let ((result (case forward-or-backward
		  (:forward
		   (gethash node-type *call-forward-features*))
		  (:backward
		   (gethash node-type *call-backward-features*))
		  (T
		   (error "cl-waffe's internal error. Features are only available when :forward and :backward")))))
    (unless result
      (nosuchnode-error "cl-waffe attempted to call ~a of ~a but couldn't find such a node. ~%Please check:~%Is the node really defined? or the dependencies are loaded correctly?" forward-or-backward model))

    (symbol-function
     (the symbol
	  (or (gethash *default-backend* result)
	      (if *restart-non-exist-backend*
		  (gethash :mgl result)
		  (restart-case
		      (error (make-condition
			      'Backend-Doesnt-Exists
			      :kernel *default-backend*
			      :node model))
		    (restart-with-mgl-kernel ()
		      (gethash :mgl result)))))))))

(defmacro get-forward-caller (model)
  "Returns the given node (model/node/optimizer)'s forward slot, which is callable with funcall/apply."
  `(get-nodefunction-caller :forward ,model))

(defmacro get-backward-caller (model)
  "Returns the given node (model/node/optimizer)'s backward slot, which is callable with funcall/apply."
  `(get-nodefunction-caller :backward ,model))

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
	  (optimizer-error "The optimizer couldn't find any parameters to optimize in the model.")
	  (butlast parameters)))))

(defmacro defoptimizer (name
			initializer-arguments
			&key
			  parameters
			  (disassemble-update nil)
			  update-declaim
			  update
			  (document "An optimizer, defined by cl-waffe."))
  "Defines optimizer in the format that cl-waffe can handle.

@begin(deflist)
@term(Name)
@def(The optimizer's structure and constructor will be defined after name)

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

;(call (SGD (find-variables model))) will works as update.
;(call-backward (SGD)) will works as zero-grads.

@end[lang=lisp](code)
"

  `(defobject ,name ,initializer-arguments
     :parameters ,parameters
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
		     (document "An node, defined by cl-waffe."))
  "Defines computation nodes in a format that cl-waffe can handle.

Note: the data structures that can be used in arguments, and returned values, must be following:

@begin(enum)
@item(WaffeTensor)
@item(1D list which each element is WaffeTensor)
@end(enum)

Be aware that you can't use (values x y ...).

@begin(deflist)
@def(name)
@term(The node's name. constructor and structure are being defined named after this argument.)

@def(initializer-argument)
@term(arguments the constructor have.)

@def(parameter)
@term(The parameters this node has being initializer with initializer-argument.)

@def(disassemble-forward)
@term(when t, when this node is compiled, display the disassemble of forward slot.)

@def(forward-declaim)
@term(Describe the declaim for the forward function. Note that the first argument is a structure. and :forward keyword in this declaim will be replaced by the forward function's name.)

@def(forward)
@term(the definition of forward)

@def(disassemble-backward)
@term(when t, when this node is compiled, display the disassemble of backward slot.)

@def(backward-declaim)
@term(Describe the declaim for the backward function. Note that the first argument is a structure. and :backward keyword in this declaim will be replaced by the backward function's name.)

@def(backward)
@term(the definition of backward)

@end(deflist)"

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

(defmacro defmodel (name
		    initializer-arguments
		    &key
		      (parameters nil)
		      (disassemble-forward nil)
		      forward-declaim
		      forward
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
     :object-type :model
     :document ,document))

(defmacro define-node-extension (name
				 &key
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

     (unless (find-class ',name nil)
       (nosuchnode-error "define-node-extension attempted to redefine the node ~a, but it doesn't exist. define-node-extension isn't evaluated the original node is defined." ',name))
     
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

     (redefine-call-inline-forward)
     (redefine-call-inline-backward)
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


(declaim (ftype (function (t) boolean) ancestor-param-p))
(defun ancestor-param-p (vars)
  ; Fixme: There must be more faster solution.
  (declare (optimize (speed 3)))
  (if (or (typecase vars
	    (list
	     (member-if #'(lambda (x)
			    (typecase x
			      (waffetensor
			       (waffetensor-is-ancestor-param x))
			      (list
			       (ancestor-param-p x))
			      (T ; defnode's argument must be consisted of waffetensor, or list consisted of waffetensor.
			       ;(error "~a" x)
			       )))
			vars))
	    (waffetensor
	     (waffetensor-is-ancestor-param vars))
	    (T
	     ;(error "~a" vars)
	     )))
      t
      nil))

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
		       (when (ancestor-param-p (list ,@,vars))
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
    (object-type
     vars
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
	       (find t ,state))
	     (is-ancestor-param
	       ,(if (eql object-type :node)
		    `(unless *no-grad*
		       (ancestor-param-p (list ,@vars)))
		    nil)))
	 ,(unless (eql object-type :node)
	    `(declare (ignore is-ancestor-param)))
	 (uncheck-node-tensor ,state ,@vars)
	 (typecase result
	   (list
	    ,(if (eql object-type :node)
		 `(map
		   'list
		   #'(lambda (x)
		       (setf (waffetensor-path-through-node? x)
			     result-next-state)
		       (unless *no-grad*
			 (progn
			   (setf (waffetensor-backward x) t)
			   (setf (waffetensor-state x) (model))
			   ; Note: variables are flat lists.
			   ; I hate this overehead of flatten...
			   (setf (waffetensor-variables x) (flatten (list ,@vars)))
			   (setf (waffetensor-is-ancestor-param x)
				 is-ancestor-param)))
		       (update-tensor-state x (flatten (list ,@vars))))
		   result)
		 `(map
		  'list
		  #'(lambda (x)
		      (setf
		       (waffetensor-path-through-node? x)
		       result-next-state)
		      x)
		  result)))
	   (T
	    ,(if (eql object-type :node)
		 `(unless *no-grad*
		    (let ((flat-vars (flatten (list ,@vars))))
		      (setf
		       (waffetensor-path-through-node? result)
		       result-next-state)
		      (setf (waffetensor-backward result) t)
		      (setf (waffetensor-state result) (model))
		      ; Note: variables are flat lists.
		      ; I hate this overehead of flatten...
		      (setf (waffetensor-variables result) flat-vars)
		      (setf (waffetensor-is-ancestor-param result)
			    is-ancestor-param)
		      result
		      (setq result (update-tensor-state result flat-vars)))
		    (let ((flat-vars (flatten (list ,@vars))))
		      (setf
		       (waffetensor-path-through-node? result)
		       result-next-state)
		      (setq result (update-tensor-state result flat-vars))))
		 `(setf
		   (waffetensor-path-through-node? result)
		   result-next-state))
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
      `(progn
	 (eval-when (:compile-toplevel
		     :load-toplevel
		     :execute)
	   (case ,forward-or-backward
	     (:forward
	      (register-forward-features ',structure-name
					 ',function-name
					 ,backend-type))
	     (:backward
	      (register-backward-features ',structure-name
					  ',function-name
					  ,backend-type))
	     (T (error "internal error"))))
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
		 ,docs
		 (locally ,@declarations
		   (with-object-macrolet-forms ',self ,args-symbols
		     ,(if (find object-type `(:node :optimizer))
			  `(with-define-function-in-defnode-way ,object-type ,args-symbols
			     ,@body)
			  `(with-define-function-in-defmodel-way ,args-symbols
			     ,@body))))))
	 (defun ,function-name (,self ,@args)
	   ,docs
	   (locally ,@declarations
	     (with-object-macrolet-forms ',self ',args-symbols
	       ,(if (find object-type `(:node :optimizer))
		    `(with-define-function-in-defnode-way ,object-type ,args-symbols
		       ,@body)
		    `(with-define-function-in-defmodel-way ,args-symbols
		       ,@body)))))
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
  "displays the given model and its parameters."
  (format stream "~%")
  (let ((*total-param-size* 0))
    (render-model-structure stream model)
    (format stream "~% -(+) Total Param: ~a" *total-param-size*)))

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


