
(in-package :cl-user)

(defpackage :lparallel-playground
  (:use :cl :lparallel :cl-waffe :mgl-mat))

(in-package :lparallel-playground)

(defmacro with-lparallel (num-core &body body)
  `(let ((lparallel:*kernel* (make-kernel ,num-core)))
     ,@body))

					; + 1 1を1000回を最適化したい並列化で

(defparameter n (* 1000 1000))

(defun test1 ()
  (dotimes (i n)
    (+ 1 1)))

(defun test2 ()
  (declare (optimize (speed 3)))
  (with-lparallel 4
    (lparallel:pdotimes (i n)
      (+ 1 1))))

(declaim (ftype (function (single-float single-float symbol) single-float) applying))
(defun applying (a b function)
  (declare
   (optimize (speed 3) (safety 0))
   (type symbol function)
   (type single-float a b))
  (case function
    (:+ (+ a b))
    (:- (- a b))
    (:* (* a b))
    (T (error "applying: function is following :+ :- :* ~a" function))))


(defun broadcasting-apply (function x y)
  (declare (optimize (speed 3) (safety 0))
	   (type symbol function)
	   (type waffetensor x y))
  ; assume that (!dims x) == (!dims y)

  (unless (= (!dims x) (!dims y))
    (error "KernelError: Can't broadcasting ~a and ~a" x y))

  (let* ((dims (cl-waffe::broadcasting x y))
	 (result-shape (loop for i fixnum upfrom 0 below (!dims x)
			     collect (let ((dim (nth i dims)))
				       (cond
					 ((and (null (car dim))
					       (null (second dim)))
					  (!shape x i))
					 ((null (car dim))
					  (!shape x i))
					 ((null (second dim))
					  (!shape y i))))))
	 (out (!zeros result-shape)))
    (declare (type list dims))

    (reshape-and-displace! (data x) `(,(!size x)) 0)
    (reshape-and-displace! (data y) `(,(!size y)) 0)
    ; Todo: CUDA Support
    (with-facets ((o  ((data out) 'backing-array :direction :output))
		  (x1 ((data x) 'backing-array :direction :input))
		  (y1 ((data y) 'backing-array :direction :input)))
      (declare (type (simple-array single-float) o x1 y1))
      (labels ((get-index (tensor index)
		 (declare (optimize (speed 3) (safety 0)))
		 (cl-waffe.backends.mgl::get-difference (data tensor) index))
	       (next (index
		      first-index-x
		      first-index-y
		      first-index-o)
		 (declare
		  (optimize (speed 3) (safety 0))
		  (type fixnum index first-index-x first-index-y first-index-o))
		 (let ((bx (car (nth index dims)))
		       (by (second (nth index dims))))
		   (when (= index (1- (length dims)))
		     ; dif=1
		     (cond
		       ((and (null bx) (null by))
			(loop for i fixnum upfrom 0 below (the fixnum (!shape x index))
			      do (setf (aref o (+ first-index-o i))
				       (applying
					(aref x1 (+ first-index-x i))
					(aref y1 (+ first-index-y i))
					function))))
		       ((null bx)
			(loop for i fixnum upfrom 0 below (the fixnum (!shape x index))
			      do (setf (aref o (+ first-index-o i))
				       (applying
					(aref x1 (+ first-index-x i))
					(aref y1 first-index-y)
					function))))
		       ((null by)
			(loop for i fixnum upfrom 0 below (the fixnum (!shape y index))
			      do (setf (aref o (+ first-index-o i))
				       (applying
					(aref x1 first-index-x)
					(aref y1 (+ first-index-y i))
					function))))
		       (T nil))
		     (return-from next nil))
		   (cond
		     ((and (null bx) (null by))
		      (let ((x-dif (get-index x index))
			    (y-dif (get-index y index))
			    (o-dif (get-index out index)))
			(declare (type fixnum x-dif y-dif o-dif))
			(if (= index 0)
			(pdotimes (i (the fixnum (!shape x index)))
			  (declare (type fixnum i))
			  (next
			   (1+ index)
			   (+ first-index-x (the fixnum (* i x-dif)))
			   (+ first-index-y (the fixnum (* i y-dif)))
			   (+ first-index-o (the fixnum (* i o-dif))))))
			(dotimes (i (the fixnum (!shape x index)))
			  (declare (type fixnum i))
			  (next
			   (1+ index)
			   (+ first-index-x (the fixnum (* i x-dif)))
			   (+ first-index-y (the fixnum (* i y-dif)))
			   (+ first-index-o (the fixnum (* i o-dif)))))))
		     ((null bx)
		      (let ((x-dif (get-index x index))
			    (o-dif (get-index out index)))
			(declare (type fixnum x-dif o-dif))
			(if (= index 0)
			(pdotimes (i by)
			  (declare (type fixnum i))
			  (next
				(1+ index)
				(+ first-index-x (the fixnum (* i x-dif)))
				first-index-y
				(+ first-index-o (the fixnum (* i o-dif))))))
			(dotimes (i by)
			  (declare (type fixnum i))
			  (next
				(1+ index)
				(+ first-index-x (the fixnum (* i x-dif)))
				first-index-y
				(+ first-index-o (the fixnum (* i o-dif)))))))
		     ((null by)
		      (let ((y-dif (get-index y index))
			    (o-dif (get-index out index)))
			(declare (type fixnum y-dif o-dif))
			(if (= index 0)
			(pdotimes (i bx)
			  (declare (type fixnum i))
			  (next
			   (1+ index)
			   first-index-x
			   (+ first-index-y (the fixnum (* i y-dif)))
			   (+ first-index-o (the fixnum (* i o-dif))))))
			(dotimes (i bx)
			  (declare (type fixnum i))
			  (next
			   (1+ index)
			   first-index-x
			   (+ first-index-y (the fixnum (* i y-dif)))
			   (+ first-index-o (the fixnum (* i o-dif)))))))
		     (T nil)))
		 nil))
	(next 0 0 0 0))
      out)))


(defun hoge1 ()
  (print 0)
  (dotimes (i 100)
    (sleep 0.01)))

(defun test ()
  (pdotimes (i 10)
    (hoge1)))

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

(defun sapply (function x y)
  ; still node debugged but it used mgl-mat's APIs
  (declare (optimize (speed 3) (safety 0))
	   (type symbol function)
	   (type waffetensor x y))
  ; assume that (!dims x) == (!dims y)
  (unless (= (!dims x) (!dims y))
    (error "KernelError: Can't broadcasting ~a and ~a" x y))

  (let* ((dims (cl-waffe::broadcasting x y))
	 (result-shape (loop for i fixnum upfrom 0 below (!dims x)
			     collect (let ((dim (nth i dims)))
				       (cond
					 ((and (null (car dim))
					       (null (second dim)))
					  (!shape x i))
					 ((null (car dim))
					  (!shape x i))
					 ((null (second dim))
					  (!shape y i))))))
	 (out (!zeros result-shape))
	 (tmp-size (last result-shape 2))
	 (x-dims-first (!shape x))
	 (y-dims-first (!shape y))
	 (x-displacement-first (mat-displacement (data x)))
	 (y-displacement-first (mat-displacement (data y))))
    (declare (type list dims x-dims-first y-dims-first))
    (reshape-and-displace! (data x) `(,(!size x)) x-displacement-first)
    (reshape-and-displace! (data y) `(,(!size y)) y-displacement-first)

    (labels ((get-stride (shape dim)
	       (let ((subscripts (fill-with-d shape dim)))
		 (apply #'+ (maplist #'(lambda (x y)
					 (the fixnum
					      (* (the fixnum (car x))
						 (the fixnum (apply #'* (cdr y))))))
					 subscripts
					 shape)))))
      (let ((x-strides (loop for i fixnum upfrom 0 below (the fixnum (length x-dims-first))
			     collect (get-stride x-dims-first i)))
	    (y-strides (loop for i fixnum upfrom 0 below (the fixnum (length y-dims-first))
			     collect (get-stride y-dims-first i)))
	    (out-strides (loop for i fixnum upfrom 0 below (the fixnum (length result-shape))
			       collect (get-stride result-shape i))))
	(labels ((x-step-index (state i repeat? dim-currently-processing)
		   (declare (type fixnum i state dim-currently-processing))
		   (+ state (if (null repeat?)
				(the fixnum (* i (the fixnum (nth dim-currently-processing x-strides))))
				0)))
		 (y-step-index (state i repeat? dim-currently-processing)
		   (declare (type fixnum i state dim-currently-processing))
		   (+ state (if (null repeat?)
				(the fixnum (* i (the fixnum (nth dim-currently-processing y-strides))))
				0)))
		 (o-step-index (state i dim-currently-processing)
		   (declare (type fixnum i state dim-currently-processing))
		   (+ state (the fixnum (* i (the fixnum (nth dim-currently-processing out-strides))))))
		 
		 (explore-batch (dims-x dims-y dims-o x-index y-index o-index dim-currently-processing)
		   (declare (type list dims-x dims-y dims-o)
			    (type fixnum x-index y-index o-index dim-currently-processing))
		   ; Parallel 3D 4D ...

		   (if (> (length dims-x) 2)
		       ; Tensor's dim >= 3, batch them until currenlt refering tensor is 2d. If *kernel*, parallelize.
		       (let* ((repeat-instruction-x (car (nth dim-currently-processing dims)))
			      (repeat-instruction-y (second (nth dim-currently-processing dims))))

			 (if (null lparallel:*kernel*)
			     (dotimes (i (nth dim-currently-processing result-shape))
			       (declare (type fixnum i))
			       (explore-batch (cdr dims-x)
					      (cdr dims-y)
					      (cdr dims-o)
					      (x-step-index x-index i repeat-instruction-x dim-currently-processing)
					      (y-step-index y-index i repeat-instruction-y dim-currently-processing)
					      (o-step-index o-index i dim-currently-processing)
					      (1+ dim-currently-processing)))
			     (lparallel:pdotimes (i (nth dim-currently-processing result-shape))
			       (declare (type fixnum i))
			       (explore-batch (cdr dims-x)
					      (cdr dims-y)
					      (cdr dims-o)
					      (x-step-index x-index i repeat-instruction-x dim-currently-processing)
					      (y-step-index y-index i repeat-instruction-y dim-currently-processing)
					      (o-step-index o-index i dim-currently-processing)
					      (1+ dim-currently-processing)))))
		       ; When processing tensors are reached to 2D/1D
		       ; Applying functions.
		       (let* ((dim-currently-processing (the fixnum (+ dim-currently-processing )))
			      (rx (car (nth dim-currently-processing dims)))
			      (ry (second (nth dim-currently-processing dims)))
			      (rx1 (car (nth (1+ dim-currently-processing) dims)))
			      (ry1 (second (nth (1+ dim-currently-processing) dims))))
			 (reshape-and-displace!
			  (data x)
			  dims-x
			  x-index)
			 (reshape-and-displace!
			  (data y)
			  dims-y
			  y-index)
			 (reshape-and-displace!
			  (data out)
			  dims-o
			  o-index)
			 (if (= (length dims-x) 1)
			     ; the rest is 1D
			     (cond
			       ((and (null rx)
				     (null ry))
				; applying the same shapes
				(case function
				      (:+
				       (copy! (data x) (data out))
				       (axpy! 1.0 (data y) (data out)))
				      (:-
				       (copy! (data x) (data out))
				       (axpy! -1.0 (data y) (data out)))
				      (:* (geem! 1.0 (data x) (data y) 0.0 (data out)))))
			       ((null rx)
				; ry is repeat (i.e: y is scalar)
				(let ((scal (mat-as-scalar (data y))))
				  (declare (type single-float scal))
				  (case function
				    (:+
				     (copy! (data x) (data out))
				     (.+! scal (data out)))
				    (:-
				     (copy! (data x) (data out))
				     (.+! (- scal) (data out)))
				    (:*
				     (axpy! scal (data x) (data out))))))
			       ((null ry)
				 ; rx is repeat (i.e: x is scalar)
				(let ((scal (mat-as-scalar (data x))))
				  (declare (type single-float scal))
				  (case function
				    (:+
				     (copy! (data y) (data out))
				     (.+! scal (data out)))
				    (:-
				     (copy! (data y) (data out))
				     (.+! (- scal) (data out)))
				    (:*
				     (axpy! scal (data y) (data out)))))))
			     ; The rest are 2D
			     (cond
			       ((and (null rx)
				     (null ry)
				     (null rx1)
				     (null ry1))
				; Shapes are the same
				; (n m) + (n m)
				(case function
				  (:+
				   (copy! (data x) (data out))
				   (axpy! 1.0 (data y) (data out)))
				  (:-
				   (copy! (data x) (data out))
				   (axpy! -1.0 (data y) (data out)))
				  (:*
				   (geem! 1.0 (data x) (data y) 0.0 (data out)))))
			       ((and
				 (or (null rx)
				     (null ry))
				 (and (null rx1)
				      (null ry1)))
				; broadcasting will be done at dim=0, dim!=1
				; iterate by columns
				; tensor is (1 m) (n m)
				(let ((row (if (null rx)
					       y
					       x))
				      (mat (if (null rx)
					       x
					       y)))
				  (case function
				    (:+
				     (fill! 1.0 (data out))
				     (scale-columns! (data row) (data out))
				     (axpy! 1.0 (data mat) (data out)))
				    (:-
				     (fill! 1.0 (data out))
				     (scale-columns! (data row) (data out))
				     (axpy! -1.0 (data mat) (data out))
				     ; x and y are reversed?
				     (if (null rx)
					 (scal! -1.0 (data out))))
				    (:*
				     (fill! 1.0 (data out))
				     (scale-columns! (data row) (data out))
				     (geem! 1.0 (data out) (data mat) 0.0 (data out))))))
			       ((and
				 (and (null rx)
				      (null ry))
				 (or  (null rx1)
				      (null ry1)))
				; broadcasting will be done at dim=1, not dim=0
				; iterate by rows
				; tensor is (n 1) (n m)
				(let ((column (if (null rx1)
					          y
					          x))
				      (mat (if (null rx1)
					       x
					       y)))
				  
				  (case function
				    (:+
				     (fill! 1.0 (data out))
				     (scale-rows! (data column) (data out))
				     (axpy! 1.0 (data mat) (data out)))
				    (:-
				     (fill! 1.0 (data out))
				     (scale-rows! (data column) (data out))
				     (axpy! -1.0 (data mat) (data out))
				     ; x and y are reversed?
				     (if (null rx1)
					 (scal! -1.0 (data out))))
				    (:*
				     (fill! 1.0 (data out))
				     (scale-rows! (data column) (data out))
				     (geem! 1.0 (data out) (data mat) 0.0 (data out))))))
			       (T
				; 2D Mat is (1 n) (m 1) or (n 1) (1 m)
				(let ((row-x (if (= (the fixnum (!shape x 0)) 1)
						 x
						 y))
				      (columns-y (if (= (the fixnum (!shape x 0)) 1)
						     y
						     x))
				      (on-the-around-way?
					(= (the fixnum (!shape x 0)) 1)))
				  (with-ones (tmp tmp-size)
				    (case function
				      (:+
				       (fill! 1.0 tmp)
				       (fill! 1.0 (data out))
				       (scale-columns! (data row-x) tmp)
				       (scale-rows! (data columns-y) (data out))
				       (axpy! 1.0 tmp (data out)))
				      (:-
				       (fill! 1.0 tmp)
				       (fill! 1.0 (data out))
				       (scale-columns! (data row-x) tmp)
				       (scale-rows! (data columns-y) (data out))
				       (axpy! -1.0 tmp (data out))
				       (when on-the-around-way?
					 (scal! -1.0 (data out))))
				      (:*
				       (fill! 1.0 (data out))
				       (scale-columns! (data row-x) (data out))
				       (scale-rows! (data columns-y) (data out)))))))))))
		   nil))
	  (explore-batch x-dims-first y-dims-first result-shape 0 0 0 0)
	  (reshape-and-displace! (data x) x-dims-first x-displacement-first)
	  (reshape-and-displace! (data y) y-dims-first y-displacement-first)
	  (reshape-and-displace! (data out) result-shape 0)
	  out)))))

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

(defun saref (out x &rest subscripts)
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
			 (cl-waffe::broadcasting
			  x
			  out)
			 nil)))
    (declare (type list x-dim-first o-dim-first))
    ; setf-mode?=t, -> the copied tensor overwrittes out
    ; setf-mode?=nil, -> creates new tensor and is overwritted

    ; (compare-dims) out>xを防ぐ
    (reshape-and-displace! (data x)   `(,(!size x)) x-displace-first)
    (reshape-and-displace! (data out) `(,(!size out)) o-displace-first)
    
    (labels ((get-stride (shape dim)
	       (let ((subscripts (loop for k fixnum upfrom 0 below dim
				       collect 0)))

		 (apply #'+ (maplist #'(lambda (x y)
					 (the fixnum
					      (* (the fixnum (car x))
						 (the fixnum (apply #'* (cdr y))))))
					 `(,@subscripts 1)
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
			     (x-step-index x-index sub dim-index)
			     (o-step-index o-index sub dim-index)))
			   (list
			    (loop for i fixnum upfrom (car sub) below (second sub)
				  do (explore-batch
				      (+ 1 dim-index)
				      (cdr dims-x)
				      (cdr dims-o)
				      (x-step-index x-index i dim-index)
				      (o-step-index o-index i dim-index))))
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

