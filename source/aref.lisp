
(in-package :cl-waffe)

(defnode ArefTensor (shape)
  :optimize t
  :parameters ((shape shape)
	       (base-shape T))
  :forward ((x) (setf (self base-shape) (!shape x))
		(apply #'!faref x (self shape))) ;thread-node??
    :backward ((dy)
	     (let ((dy-n (!zeros (self base-shape))))
	       (setf (!areflist dy-n (self shape)) dy)
	       (list dy-n))))

(defnode SetfArefTensor (shape)
  :optimize t
  :parameters ((shape shape))
  :forward ((x y)
	    ; Note: defnode must produce new sysconst otherwise stackoverflow...
	    (sysconst (data (apply #'!write-faref x y (self shape)))))
  :backward ((dy)
	     (list dy (apply #'!faref dy (self shape)))))

(defun saref-p (setf-mode? out tensor subscripts broadcasts)
  (declare (ignore broadcasts)
	   (optimize (speed 3) (safety 0)))
  (let ((topic-tensor (if setf-mode?
			  out
			  tensor)))
    (declare (type waffetensor topic-tensor))
    (mapc
     #'(lambda (x y)
	 (typecase y
	   (fixnum
	    (if (>= y x)
		(error "!aref the index ~a is beyonds ~a.~%~a~%and~%~a~%~% dims are specified in the range of (0 ~a)" y x topic-tensor subscripts (1- (the fixnum x)))))
	   (list
	    (if (> (the fixnum (car y)) (the fixnum x))
		(error "!aref the first index ~a is beyonds ~a.~%~a~%and~%~a~%~% dims are specified in the range of (0 ~a)" y x topic-tensor subscripts (1- x)))
	    (if (> (the fixnum (second y)) x)
		(error "!aref the second index ~a is beyonds ~a.~%~a~%and~%~a~%~% stops are specified in the range of (0 ~a)" y x topic-tensor subscripts (1- x))))))
     (!shape topic-tensor) subscripts))
  (mapc
   #'(lambda (a b c)
       (typecase b
	 (fixnum t)
	 (cons (if (< (the fixnum a) (the fixnum (- (the fixnum (second b)) (the fixnum (car b)))))
		 (error "!aref: Can't copy ~%~a and ~%~a ~%~%beacuse the subscript ~a will produce the tensor whose shape is (~a)~% but it won't fit into the tensor of (~a)" out tensor b (the fixnum (- (the fixnum (second b)) (the fixnum (car b)))) a)))
	 (t (if (< (the fixnum a) (the fixnum c))
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
  ; Todo: Add handler condition because reshaped tensor won't fixed.
  ;(apply #'%faref tensor dims)
  (apply #'%saref nil tensor dims))

; wrapper
(defun !write-faref (tensor value &rest dims)
  "Overwrites to tensor, with reading value of dims."
  (unless (= (!dims value) (!dims tensor))
    (error "!write-faref: the size of dim doesn't match. use !unsqueeze and !squeeze to adjust it.: ~a and ~a" (!dims value) (!dims tensor)))
  ;(apply #'%write-faref tensor value dims)
  (apply #'%saref tensor value dims))

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


(declaim (ftype (function (cons fixnum) cons) fill-with-d))
(defun fill-with-d (shape i)
  (declare (optimize (speed 3))
	   (type cons shape)
	   (type fixnum i))
  (let ((index -1))
    (declare (type fixnum index))
    (map 'list (lambda (x)
		 (declare (ignore x))
		 (incf index 1)
		 (cond
		   ((= i index)
		    1)
		   (T 0)))
	 shape)))

; broadcasting for saref
(defun broadcasting1 (x y)
  "returns broadcasting instructions but won't return error"
  (declare (type waffetensor x y))
  (map 'list #'(lambda (xi yi)
		 (declare (type fixnum xi yi))
		 (if (and (or (= xi 1) (= yi 1))
			  (not (= xi yi)))
		     (if (= xi 1)
			 `(,(max xi yi) nil)
			 `(nil ,(max xi yi)))
		     (if (= xi yi)
			 `(nil nil)
			 `(nil nil))))
       (!shape x) (!shape y)))

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
			 (broadcasting1
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
			      with 1+dim-index fixnum = (1+ dim-index)
			      for i fixnum upfrom 0 below (- (the fixnum (second sub)) (the fixnum (car sub)))
				  do (explore-batch
				      1+dim-index
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
			    (loop with 1+dim-index fixnum = (1+ dim-index)
			          for i fixnum upfrom 0 below (car dims-o)
				  do (explore-batch
				      1+dim-index
				      (cdr dims-x)
				      (cdr dims-o)
				      (x-step-index
				       x-index
				       (mod i (the fixnum (nth dim-index x-dim-first))) ; out of range.
				       dim-index)
				      (o-step-index
				       o-index
				       (mod i (the fixnum (nth dim-index o-dim-first))) ; out of range.
				       dim-index))))))
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
			 (if (same-shape-p x out)
			     ; if the last dim is not repeating...
			     (copy! (data x) (data out))
			     ; otherwise (the last dims should be repeated...)
			     (if (= (the fixnum (!shape x 0)) 1)
				 (fill! (mat-as-scalar (data x)) (data out))
				 (let ((stride (the fixnum (!shape x 0))))
				   (loop for k fixnum upfrom 0 below (the fixnum (!shape out 0)) by stride
					 do (progn
					      (reshape-and-displace!
					       (data out)
					       x-size
					       (the fixnum (+ o-begin o-index k)))					      
					 (copy! (data x)
						(data out)))))))))
		   nil))
	  (explore-batch 0 x-dim-first o-dim-first 0 0)
	  (reshape-and-displace! (data x) x-dim-first x-displace-first)
	  (reshape-and-displace! (data out) o-dim-first o-displace-first)
	  out)))))
