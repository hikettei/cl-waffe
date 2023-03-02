
(in-package :cl-waffe)

#|
Here's Mathematical Functions and Utils:
  1.Model-List
  2.!aref/(setf !aref)
|#

(defparameter pi-single-float (the single-float (coerce pi 'single-float)))

(defnode ReLUTensor nil
  :optimize t
  :parameters ((path-through nil) (zero-buff nil))
  :forward ((x) ; Todo rewrite more faster way.
		(unless (self zero-buff)
		  (setf (self zero-buff) (!zeros (!shape x))))
		(let ((mask (with-searching-calc-node :< x (self zero-buff))))
		  (save-for-backward path-through mask)
		  (!mul mask x)))
  :backward ((dy)
	     (list (!mul (self path-through) dy))))

(defun !relu (x)
  "Applying relu to x, return a new sysconst with making nodes.

Relu(x) = { 0 (x < 0), x (x > 0) }

Input: x where x is waffe supported data type.

Output: Tensor"
  (call (ReLUTensor) (assure-tensor x)))

;(defun !relu1 (x)
;  (!mul x (!where #'(lambda (v) (>= v 0.0)) x 1.0 0.0)))

(defnode SigmoidTensor nil
  :optimize t
  :parameters ((xi T))
  :forward ((x)
	    (save-for-backward xi x)
            (!div (!add 1 (!tanh (!div x 2)))
		  (const 2)))
  :backward ((dy) (let ((p (!sigmoid (self xi))))
		    (list (!mul p (!mul dy (!sub 1 p)))))))

(defun !sigmoid (x)
  "Applyong sigmoid to x, return a new sysconst with making nodes.

Input: x where x is waffe supported data type.

Output: Tensor"
  (call (SigmoidTensor) (assure-tensor x)))

(defnode TanhTensor nil
  :optimize t
  :parameters ((xi T))
  :forward ((x)
	    (save-for-backward xi x)
	    (with-searching-calc-node :tanh x))
  :backward ((dy)
	     (list (!mul dy (!sub (const 1) (!pow (!tanh (self xi)) 2))))))

(defun !tanh (x)
  "Applying tanh to x, return a new sysconst with making nodes."
  (call (TanhTensor) (assure-tensor x)))

; Optimizing won't go well
(defun !gelu (x &key (approximate t))
  "Applying gelu to x, returning a new sysconst.

Paper: https://arxiv.org/abs/1606.08415.

TOOD: Improve its performance

GeLU(x) = x * s(x)

When approximate is t:

s(x) = x/2 * [1 + tanh(sqrt(2/pi * (x + 0.044715 * x^3)))]

When is nil:

Not implemented (TODO)

@begin[lang=lisp](code)
(setq x (!randn `(10 10)))
(!gelu x)
;#Const(((0.201... 0.038... ~ 0.158... 0.040...)        
;                 ...
;        (0.300... 1.395... ~ 0.030... 0.029...)) :mgl t :shape (10 10))
@end[lang=lisp](code)"
  (declare (optimize (speed 3)))
  ; s(x) is not necessary derivable ↓ is is-ancestor-tensor considered?
  (let ((s (if approximate
	       (!mul (!div x 2)
		     (!filter x
			      #'(lambda (el)
				  (declare (type single-float el))
				  (multiple-value-bind (n)
				      ; failed to optimize
				      (floor
				       (the single-float
					    (* (sqrt (the (single-float 0e0)
							  (/ 2.0
							     (the single-float pi-single-float))))
					       (+ el
						  (* 0.044715
						     (expt el 3))))))
				    (the single-float (+ 1.0 (tanh n)))))))
	       (error "no implemented yet"))))
    (!mul x s)))

(defun !leakey-relu (x &optional (alpha 0.01))
  "Applying Leakey-relu to x, returning a new sysconst.

Leakey-ReLU is defined as out = {alpha (x < 0), x (x >= 0)}

Example:

@begin[lang=lisp](code)
(setq x (!randn `(10 10)))
#Const(((0.635... -0.56... ~ -1.15... -1.50...)        
                 ...
        (0.775... 1.258... ~ -1.29... 0.240...)) :mgl t :shape (10 10))

(!leakey-relu x)
#Const(((0.635... 0.003... ~ 0.013... 0.022...)        
                 ...
        (0.775... 1.258... ~ 0.016... 0.240...)) :mgl t :shape (10 10))
@end[lang=lisp](code)"
  (declare (optimize (speed 3))
	   (type single-float alpha))
  (!mul x (!filter x #'(lambda (x)
			 (declare (type single-float x))
			 (if (>= x 0)
			     1.0
			     (* alpha x))))))

(defmodel Swish (&key (beta 1.0)
		      (trainable t))
  :parameters ((beta (if trainable
			 (tensor beta)
			 (const beta))))
  :forward ((x)
	    (!swish x :beta (self beta))))

(defun !swish (x &key (beta (const 1.0)))
  "Applying swish to each element of x

Swish is defined as out = (/ 1 (+ 1 (exp (* beta -1 x))))

In default beta is 1.0, if you want to use trainable one, @cl:param(Swish) is available as a waffe model.

Note that beta must begin given as a waffetensor.

@begin[lang=lisp](code)
(setq x (!randn `(10 10)))
#Const(((0.635... -0.56... ~ -1.15... -1.50...)        
                 ...
        (0.775... 1.258... ~ -1.29... 0.240...)) :mgl t :shape (10 10))

(!swish x)
;#Const(((0.415... -0.20... ~ -0.27... -0.27...)        
;                 ...
;        (0.531... 0.980... ~ -0.27... 0.134...)) :mgl t :shape (10 10))

(call (Swish :beta 1.0) x) ; its beta is trainable by backpropgating.
;#Const(((0.415... -0.20... ~ -0.27... -0.27...)        
;                 ...
;        (0.531... 0.980... ~ -0.27... 0.134...)) :mgl t :shape (10 10))
@end[lang=lisp](code)"
  (!div x (!add 1 (!exp (!mul (!mul -1 beta) x)))))

(defun !mish () "Todo")

(defun !average (x)
  (let ((z (!sum x 1))
	(batch-size (!shape x 0)))
    (!!div z batch-size))) ; slow?

(defun !softmax-function (x &key (avoid-overflow t))
  "Applying softmax.

!softmax has three behaivour depending on the number of dimensions."
  (declare (optimize (speed 3))
	   (type waffetensor x))
  (case (!dims x)
    (1 (!softmax-function (!unsqueeze x)))
    (2 (let* ((x1 (if avoid-overflow
		      (!sub x (!average x))
		      x))
	      (z (!sum (!exp x1) 1)))
	 (!!div (!exp x1) z)))
    (3 (let* ((result (!zeros (!shape x)))) ; For batched inputs
	 (dotimes (i (the fixnum (!shape x 0)))
	   (setq result (setf (!aref result i)
			      (!softmax-function (!squeeze (!aref x i) 0)))))
	 result))
    (T (error "!softmax: Not implemented. softmax only supports where (!dims tensor) <= 3."))))

(defmodel SoftMaxNode (avoid-overflow)
  :parameters ((avoid-overflow avoid-overflow))
  :forward ((x)
	    (!softmax-function x :avoid-overflow (self avoid-overflow))))

(defun !softmax (x &key (avoid-overflow t))
  "Applying softmax to x. !softmax has three behaviours depending on the number of dimensions.

The number of dims is...
@begin(deflist)
@def(1)
@begin(term)
Softmax is applied to dim=0
@begin[lang=lisp](code)
(setq a (!randn `(10)))
(!softmax a)
;#Const((0.910... 0.886... ~ 0.802... 0.616...) :mgl t :shape (10))
@end[lang=lisp](code)
@end(term)

@def(2)
@begin(term)
Softmax is applied to dim=0
@begin[lang=lisp](code)
(setq a (!randn `(10 10)))
;#Const(((-0.29... -1.99... ~ -0.36... 1.725...)        
;                 ...
;        (0.695... -0.94... ~ 1.179... 0.655...)) :mgl t :shape (10 10))

(!softmax a)
;#Const(((0.064... 0.011... ~ 0.060... 0.489...)        
;                 ...
;        (0.129... 0.024... ~ 0.209... 0.124...)) :mgl t :shape (10 10))
@end[lang=lisp](code)
@end(term)

@def(3)
@begin(term)
Softmax is applied to dim=0
@begin[lang=lisp](code)
(setq a (!randn `(10 10 10)))
;#Const((((2.585... 0.517... ~ 0.428... 0.059...)         
;                   ...
;         (-2.11... 0.308... ~ -0.91... 0.649...))        
;                 ...
;        ((-0.75... 1.030... ~ 0.656... -0.00...)         
;                   ...
;         (-0.37... -0.52... ~ 1.589... -0.10...))) :mgl t :shape (10 10 10))

(!softmax a)
;#Const((((0.374... 0.047... ~ 0.043... 0.029...)         
;                   ...
;         (0.010... 0.115... ~ 0.033... 0.162...))        
;                 ...
;        ((0.029... 0.172... ~ 0.118... 0.061...)         
;                   ...
;         (0.048... 0.041... ~ 0.345... 0.063...))) :mgl t :shape (10 10 10))
@end[lang=lisp](code)
@end(term)

@def(4)
@begin(term)
Todo: currently, it returns error.
@begin[lang=lisp](code)
@end[lang=lisp](code)
@end(term)
@end(deflist)"
  (declare (type waffetensor x))
  (call (SoftMaxNode avoid-overflow) x))

; Model Lists

(defmodel model-list (model-list)
  :document (with-usage "model-list"
	      :overview "define model sequentially, (e.g. x = (sequence `((layer1) (layer2))), (call x 1 tensor) => layer1's output)"
	      :args "model1 model2 ..."
	      :forward "@cl:param(index) represents the index of models. @cl:param(args) is the arguments for index-th model."
	      :step-args "index &rest args")
  :parameters ((mlist model-list))
  :forward ((index &rest args)
	    (error "model-list couldn't pass call correctly")))

(defun mlist (&rest models)
  "define mlist"
  (model-list models))

(defun mth (index mlist)
  "Accessor for model-list"
  (declare (type model-list mlist))
  (nth (typecase index
	 (waffetensor (data index))
	 (T index))
       (model-list-mlist mlist)))


; !aref, (setf !aref) function and nodes:

(defnode ArefTensor (shape)
  :regard-as-node nil
  :parameters ((shape shape)
	       (base-shape T))

  :forward ((x) (setf (self base-shape) (!shape x))
		(apply #'!faref x (self shape))) ;thread-node??
  :backward ((dy)
	     (let ((dy-n (!zeros (self base-shape))))
	       (setf (!areflist dy-n (self shape)) dy)
	       (list dy-n))))

(defnode SetfArefTensor (shape)
  :parameters ((shape shape))
  :regard-as-node nil
  :forward ((x y)
	    ; Note: defnode must produce new sysconst otherwise stackoverflow...
	    (sysconst (data (apply #'!write-faref x y (self shape)))))
  :backward ((dy)
	     (list dy (apply #'!faref dy (self shape)))))


#|
For those who are reading my ugly code.
There's two !aref:
  1. %faref/%write-faref (old implementation using facets)
  2. %saref (new implementation using BLAS Operations
%faref %write-faref is not used anymore but it supports simple-array.
I set aside %faref for testing %saref

Note: !aref/(setf !aref) definitions are located at tensor.lisp
|#

; wrapper
(defun !faref (tensor &rest dims)
  (value tensor)
  ;(apply #'%faref tensor dims)
  (apply #'%saref nil tensor dims))

; wrapper
(defun !write-faref (tensor value &rest dims)
  (unless (= (!dims value) (!dims tensor))
    (error "!write-faref: the size of dim doesn't match. use !unsqueeze and !squeeze to adjust it.: ~a and ~a" (!dims value) (!dims tensor)))
  (apply #'%write-faref tensor value dims)
  ;(apply #'%saref tensor value dims)
  )

(defun %faref (tensor &rest dims)
  (declare (optimize (speed 3))
	   (type waffetensor tensor))
  (value tensor)
  (let* ((tensor-dims (!shape tensor))
	 (dims (cond
		 ((> (!dims tensor) (length dims))
					; adjust the size of dim
		  (concatenate
		   'list
		   dims
		   (repeat-n t (the fixnum (- (!dims tensor)
					      (the fixnum
						   (length dims)))))))
		 ((= (!dims tensor) (length dims)) dims)
		 (T
		  (error "!aref: dim ~a beyonds tensor's dim ~a"
			 dims
			 (!shape tensor)))))
	 (dims (loop for i fixnum upfrom 0 below (length dims)
		     collect (let ((x (nth i dims)))
			       (typecase x
				 (fixnum
				  (if (< x 0)
				      (the fixnum
					   (+ (the fixnum (!shape tensor i))
					      (the fixnum x)))
				      x))
				 (list
				  (list
				   (if (< (the fixnum (car x)) 0)
				       (the fixnum
					    (+
					     (the fixnum (!shape tensor i))
					     (the fixnum (car x))))
				       (car x))
				   (if (<= (the fixnum (second x)) 0)
				       (the fixnum
					    (+
					     (the fixnum (!shape tensor i))
					     (the fixnum (second x))))
				       (second x))))
				 (T x)))))
	 (dims-result
	   (mapcar
	    #'(lambda (x y)
		(typecase x
		  (fixnum 1)
		  (list
		   (unless (= (length x) 2)
		     (error "!faref: the range is specified by list, but length != 2. at ~a" dims))
		   (the fixnum (- (the fixnum (second x))
				  (the fixnum (first x)))))
		  (T y)))
	    dims tensor-dims))
	 (dims-indices
	   (mapcar #'(lambda (x y)
		       (typecase x
			 (fixnum 1)
			 (list (repeat-c (the fixnum
					      (- (the fixnum (second x))
						 (the fixnum (car x))))
					 :start (car x)))
			 (T (repeat-c y))))
		   dims dims-result))
	 (result (!zeros dims-result)))

    (loop for i fixnum upfrom 0 below (length dims)
	  do (let ((x (nth i dims)))
	       (typecase x
		 (fixnum
		  (if (or (< x 0)
			  (> x (the fixnum (!shape tensor i))))
		      (error "!faref: dims must be in the range of 0<=x<=~a, but got ~a. when processing ~a, called with ~a" (!shape tensor i) x tensor dims)))
		 (list
		  (if (or (< (the fixnum (car x)) 0)
			  (> (the fixnum (car x))
			     (the fixnum (!shape tensor i))))
		      (error "!faref: dims must be in the range of 0<=x<=~a, but got ~a. when processing ~a, called with ~a" (!shape tensor i) (car x) tensor dims))
		  (if (or (< (the fixnum (second x)) 0)
			  (> (the fixnum (second x))
			     (the fixnum (!shape tensor i))))
		      (error "!faref: dims must be in the range of 0<=x<=~a, but got ~a. when processing ~a, called with ~a" (!shape tensor i) (second x) tensor dims))))))
    
    (with-facets ((from-array   ((data tensor) 'array :direction :input))
		  (result-array ((data result) 'array :direction :output)))
      (labels ((next-node (drest args rargs)
		 (if (= (length args) (length dims))
		     (apply
		      #'(setf aref)
		      (apply #'aref from-array args)
		      result-array
		      rargs))

		 (if (typep (car drest) 'fixnum)
		     (next-node
		      (cdr drest)
		      (concatenate 'list args
				   `(,(nth (length args) dims)))
		      (concatenate 'list rargs `(0)))

		     (loop
		       for i fixnum
		       upfrom 0
			 below (length (the list (car drest)))
		       do (next-node (cdr drest)
				     (concatenate
				      'list
				      args
				      `(,(nth i (car drest))))
				     (concatenate
				      'list
				      rargs
				      `(,i)))))))
	(next-node dims-indices nil nil)
	result))))

(defun %write-faref (tensor value &rest dims)
  "(setf tensor value)

(!aref tensor dims) <- (!aref value (!shape dims))"
  (declare (optimize (speed 3))
	   (type waffetensor tensor))
  (unless (= (!dims value) (!dims tensor))
    (error "!write-faref: the size of dim doesn't match. use !unsqueeze and !squeeze to adjust it.: ~a and ~a" (!dims value) (!dims tensor)))
  (value value)
  (value tensor)
  (let* ((tensor-dims (!shape tensor))
	 (dims (cond
		 ((> (!dims tensor) (length dims))
		  (concatenate
		   'list
		   dims
		   (repeat-n t (the fixnum (- (!dims tensor)
					      (the fixnum
						   (length dims)))))))
		 ((= (!dims tensor) (length dims)) dims)
		 (T
		  (error "!aref: dim ~a beyonds tensor's dim ~a"
			 dims
			 (!shape tensor)))))
	 (dims (loop for i fixnum upfrom 0 below (length dims)
		     collect (let ((x (nth i dims)))
			       (typecase x
				 (fixnum
				  (if (< x 0)
				      (the fixnum
					   (+ (the fixnum (!shape tensor i))
					      (the fixnum x)))
				      x))
				 (list
				  (list
				   (if (< (the fixnum (car x)) 0)
				       (the fixnum
					    (+
					     (the fixnum (!shape tensor i))
					     (the fixnum (car x))))
				       (car x))
				   (if (<= (the fixnum (second x)) 0)
				       (the fixnum
					    (+
					     (the fixnum (!shape tensor i))
					     (the fixnum (second x))))
				       (second x))))
				 (T x)))))
	 (dims-result
	   (mapcar
	    #'(lambda (x y)
		(typecase x
		  (fixnum 1)
		  (list
		   (unless (= (length x) 2)
		     (error "!faref: the range is specified by list, but length != 2. at ~a" dims))
		   (the fixnum (- (the fixnum (second x))
				  (the fixnum (first x)))))
		  (T y)))
	    dims tensor-dims))
	 (dims-indices
	   (mapcar #'(lambda (x y)
		       (typecase x
			 (fixnum 1)
			 (list (repeat-c (the fixnum
					      (- (the fixnum (second x))
						 (the fixnum (car x))))
					 :start (car x)))
			 (T (repeat-c y))))
		   dims dims-result))
	 (reshaped-tensor value))
    (loop for i fixnum upfrom 0 below (length dims)
	  do (let ((x (nth i dims)))
	       (typecase x
		 (fixnum
		  (if (or (< x 0)
			  (> x (the fixnum (!shape tensor i))))
		      (error "!faref: dims must be in the range of 0<=x<=~a, but got ~a. when processing ~a, called with ~a" (!shape tensor i) x tensor dims)))
		 (list
		  (if (or (< (the fixnum (car x)) 0)
			  (> (the fixnum (car x))
			     (the fixnum (!shape tensor i))))
		      (error "!faref: dims must be in the range of 0<=x<=~a, but got ~a. when processing ~a, called with ~a" (!shape tensor i) (car x) tensor dims))
		  (if (or (< (the fixnum (second x)) 0)
			  (> (the fixnum (second x))
			     (the fixnum (!shape tensor i))))
		      (error "!faref: dims must be in the range of 0<=x<=~a, but got ~a. when processing ~a, called with ~a" (!shape tensor i) (second x) tensor dims))))))

    (with-facets ((from-array ((data reshaped-tensor) ; todo for cuda.
			       'array :direction :input))
		  (result-array ((data tensor)
				 'array :direction :output)))
      (labels ((next-node (drest args rargs)
		 (if (= (length args) (length dims))
		     (apply
		      #'(setf aref)
		      (apply
		       #'aref
		       from-array
		       (loop for i fixnum upfrom 0 below (length rargs)
			     collect (mod
				      (the fixnum (nth i rargs))
				      (the fixnum (!shape value i)))))
		      result-array
		      args))

		 (if (typep (car drest) 'fixnum)
		     (next-node
		      (cdr drest)
		      (concatenate 'list args
				   `(,(nth (length args) dims)))
		      (concatenate 'list rargs `(0)))
		     (loop
		       for i fixnum
		       upfrom 0
			 below (length (the list (car drest)))
		       do (next-node (cdr drest)
				     (concatenate
				      'list
				      args
				      `(,(nth i (car drest))))
				     (concatenate
				      'list
				      rargs
				      `(,i)))))))
	(next-node dims-indices nil nil)
	tensor))))

(defun fill-with-d (shape i)
  (let ((index -1))
    (map 'list (lambda (x)
		 (declare (ignore x))
		 (incf index 1)
		 (cond
		   ((= i index)
		    1)
		   (T 0)))
	 shape)))

(defun saref-p (setf-mode? out tensor subscripts broadcasts)
  (declare (ignore broadcasts))
  (let ((topic-tensor (if setf-mode?
			  out
			  tensor)))
    (mapc
     #'(lambda (x y)
	 (typecase y
	   (fixnum
	    (if (>= y x)
		(error "!aref the index ~a is beyonds ~a.~%~a~%and~%~a~%~% dims are specified in the range of (0 ~a)" y x topic-tensor subscripts (1- x))))
	   (list
	    (if (> (car y) x)
		(error "!aref the first index ~a is beyonds ~a.~%~a~%and~%~a~%~% dims are specified in the range of (0 ~a)" y x topic-tensor subscripts (1- x)))
	    (if (> (second y) x)
		(error "!aref the second index ~a is beyonds ~a.~%~a~%and~%~a~%~% stops are specified in the range of (0 ~a)" y x topic-tensor subscripts (1- x))))))
     (!shape topic-tensor) subscripts))
  (mapc
   #'(lambda (a b c)
       (typecase b
	 (fixnum t)
	 (cons (if (< a (- (second b) (car b)))
		 (error "!aref: Can't copy ~%~a and ~%~a ~%~%beacuse the subscript ~a will produce the tensor whose shape is (~a)~% but it won't fit into the tensor of (~a)" out tensor b (- (second b) (car b)) a)))
	 (t (if (< a c)
		(error "!aref: Can't copy ~a and ~a ~%~%because the size ~a tensor won't fit into the size ~a tensor.~%That is, the given out is too small to copy the target.~%~%(setf (!aref out subscripts) target) <- out is too small." out tensor c a)))))
   (!shape out) subscripts (!shape tensor)))

(defun parse-subscripts (tensor subscripts)
  (declare (optimize (speed 3))
	   (type waffetensor tensor)
	   (type (or null cons) subscripts))
  (unless (>= (!dims tensor) (length subscripts))
    (error "!aref: The number of subscripts is larger than given tensor: ~a and ~a" (!shape tensor) subscripts))
  (loop for i fixnum upfrom 0 below (!dims tensor)
	collect (let ((subscript (or (nth i subscripts) t))
		      (shape (!shape tensor i)))
		  (declare (type fixnum shape))
		  (typecase subscript
		    (fixnum
		     (if (>= subscript 0)
			 subscript
			 (the fixnum (+ shape (the fixnum subscript)))))
		    (list
		     (unless (= (length subscript) 2)
		       (error "!aref: The subscript is invaild: Subscripts are given by '(start stop) but got ~a." subscripts))
		     (let ((subscript (map 'list #'(lambda (a)
						     (typecase a
						       (fixnum
							(if (>= a 0)
							    a
							    (the fixnum (+ shape a))))
						       (t
							(unless (eql a t)
							  (error "!aref: The subscript is invaild: cons arguments are given by fixnum but got ~a, at ~a" a subscript))
							shape)))
					   subscript)))

		       (unless (< (the fixnum (car subscript)) (the fixnum (second subscript)))
			 (error "!aref: The subscript is invaild: Got ~a but the first argument is larget than the second one." subscript))
		       subscript))
		    (t
		     (unless (eql subscript t)
		       (error "!aref: The format is invaild. Subscripts are given by following format: fixnum cons t but got ~a" subscript))
		     t)))))

(defun %saref (out x &rest subscripts)
  "saref excepts to be out and x's dims are the same."
  (declare (optimize (speed 3))
	   (type cons subscripts)
	   (type waffetensor x)
	   (type (or null waffetensor) out))
  (let* ((setf-mode? (not (null out)))
	 (subscripts (parse-subscripts (if setf-mode?
					   out
					   x)
				       subscripts))
	 (out-shape (or (if (not (null out))
			    (!shape out))
			(loop for i fixnum upfrom 0 below (length (the list subscripts))
			      collect (let ((sub (nth i subscripts)))
					(typecase sub
					  (fixnum 1)
					  (cons (the fixnum (- (the fixnum (second sub))
							       (the fixnum (car sub)))))
					  (T (!shape x i)))))))
	 (out (or out
		  (!zeros out-shape)))
	 (x-dim-first (!shape x))
	 (o-dim-first (!shape out))
	 (x-displace-first (mat-displacement (data x)))
	 (o-displace-first (mat-displacement (data out)))
	 (broadcasts (if setf-mode?
			 (broadcasting
			  x
			  out)
			 nil)))
    (declare (type list x-dim-first o-dim-first))
    #|
    setf-mode?=t, -> the copied tensor overwrittes out
    setf-mode?=nil, -> creates new tensor and is overwritted
    and when setf-mode is t, subscriptions affect outs.
    |#

    (saref-p
     setf-mode?
     out
     x
     subscripts
     broadcasts)

    (reshape-and-displace! (data x)   `(,(!size x)) x-displace-first)
    (reshape-and-displace! (data out) `(,(!size out)) o-displace-first)
    
    (labels ((get-stride (shape dim)
	       (let ((subscripts (fill-with-d shape dim)))
		 (apply #'+ (maplist #'(lambda (x y)
					 (the fixnum
					      (* (the fixnum (car x))
						 (the fixnum (apply #'* (cdr y))))))
				     subscripts
				     shape)))))

      (let ((x-strides (loop for i fixnum upfrom 0 below (the fixnum (length x-dim-first))
			     collect (get-stride x-dim-first i)))
	    (o-strides (loop for i fixnum upfrom 0 below (the fixnum (length o-dim-first))
			     collect (get-stride o-dim-first i))))
	(labels ((x-step-index (state i dim-index)
		   (declare (type fixnum state i dim-index))
		   (if (or (null broadcasts)
			   (null (car (nth dim-index broadcasts))))
		       (+ state (the fixnum
				     (* i (the fixnum (nth dim-index x-strides)))))
		       state))
		 (o-step-index (state i dim-index)
		   (declare (type fixnum state i dim-index))
		   (if (or (null broadcasts)
			   (null (second (nth dim-index broadcasts))))
		       (+ state (the fixnum
				     (* i (the fixnum (nth dim-index o-strides)))))
		       state))
		 (explore-batch (dim-index dims-x dims-o x-index o-index)
		   (declare (type fixnum dim-index x-index o-index)
			    (type list dims-x dims-o))
		   (if (>= (length dims-x) 2)
		       ; Batch
		       (let ((sub (nth dim-index subscripts)))
			 (typecase sub
			   (fixnum
			    (explore-batch
			     (+ 1 dim-index)
			     (cdr dims-x)
			     (cdr dims-o)
			     (x-step-index
			      x-index
			      (if setf-mode?
				  0
				  sub)
			      dim-index)
			     (o-step-index
			      o-index
			      (if setf-mode?
				  sub
				  0)
			      dim-index)))
			   (list
			    (loop
			      with m fixnum = (car sub)
			      for i fixnum upfrom 0 below (- (the fixnum (second sub)) (the fixnum (car sub)))
				  do (explore-batch
				      (+ 1 dim-index)
				      (cdr dims-x)
				      (cdr dims-o)
				      (x-step-index
				       x-index
				       (if setf-mode?
					   i
					   (+ m i))
				       dim-index)
				      (o-step-index
				       o-index
				       (if setf-mode?
					   (+ m i)
					   i)
				       dim-index))))
			   (t
			    (loop for i fixnum upfrom 0 below (car dims-o)
				  do (explore-batch
				      (+ 1 dim-index)
				      (cdr dims-x)
				      (cdr dims-o)
				      (x-step-index x-index i dim-index)
				      (o-step-index o-index i dim-index))))))
					; Apply copy
		       (let* ((sub (nth dim-index subscripts))
			      (x-size (if setf-mode?
					  dims-x
					  (typecase sub
					    (fixnum `(1))
					    (cons `(,(the fixnum (- (the fixnum (second sub)) (the fixnum (car sub))))))
					    (t dims-x))))
			      (o-size (if setf-mode?
					  (typecase sub
					    (fixnum `(1))
					    (cons `(,(the fixnum (- (the fixnum (second sub)) (the fixnum (car sub))))))
					    (t dims-o))
					  dims-o))
			      (x-begin (if setf-mode?
					   0
				           (typecase sub
					     (fixnum sub)
					     (cons (car sub))
					     (t 0))))
			      (o-begin (if setf-mode?
					   (typecase sub
					     (fixnum sub)
					     (cons (car sub))
					     (t 0))
					   0)))
			 (declare (type fixnum x-begin o-begin))
			 (reshape-and-displace!
			  (data x)
			  x-size
			  (the fixnum (+ x-begin x-index)))
			 (reshape-and-displace!
			  (data out)
			  o-size
			  (the fixnum (+ o-begin o-index)))
			 (copy! (data x) (data out))))
		   nil))
	  (explore-batch 0 x-dim-first o-dim-first 0 0)
	  (reshape-and-displace! (data x) x-dim-first x-displace-first)
	  (reshape-and-displace! (data out) o-dim-first o-displace-first)
	  out)))))
