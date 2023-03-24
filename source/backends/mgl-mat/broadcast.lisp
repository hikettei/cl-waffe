
(in-package :cl-waffe.backends.mgl)


(defparameter *use-blas-min-size* 100 "This thereshold decides what functions to use to broadcast. If the product of the last two dimensions is above this threshold, broadcast with a blas instruction.")

#|
For people who is reading my ugly code:

There's two broadcasting functions.
  1. broadcasting-apply-facet
  2. broadcasting-apply-mgl

The first one uses with-facet, on the other hand the second one uses BLAS/CUDA Operations.

When cuda is enabled or the last two dims are larger than *use-blas-broadcast-min*, using the second one.

However, unless, use the first one because using BLAS Operations is a little extravagance for small matrixs.

These function are called by broadcasting-apply
|#

(defun broadcasting-apply (function x y)
  ; consider use-cuda-p?
  (declare (optimize (speed 3)))
  (let ((last-dims-x (apply #'* (last (!shape x) 2)))
	(last-dims-y (apply #'* (last (!shape y) 2))))
    (declare (type fixnum last-dims-x last-dims-y))
    (if (>= (min last-dims-x last-dims-y) (the fixnum *use-blas-min-size*))
	(broadcasting-apply-mgl   function x y)
	(broadcasting-apply-facet function x y))))

(declaim (ftype (function (single-float single-float symbol) single-float) applying))
(defun applying (a b function)
  "Broadcasting only supports single-float"
  (declare
   (optimize (speed 3) (safety 0))
   (type symbol function)
   (type single-float a b))
  (case function
    (:+ (+ a b))
    (:- (- a b))
    (:* (* a b))
    (T (error "applying: function is following :+ :- :* ~a" function))))

(defun broadcasting-apply-facet (function x y)
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
    ; Todo: CUDA Support lparallel
    (with-facets ((o  ((data out) 'backing-array :direction :output))
		  (x1 ((data x) 'backing-array :direction :input))
		  (y1 ((data y) 'backing-array :direction :input)))
      (declare (type (simple-array single-float) o x1 y1))
      
      (labels ((get-index (tensor index)
		 (declare (optimize (speed 3) (safety 0))
			  (inline get-difference))
		 (get-difference (data tensor) index))
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
		      (loop with x-dif fixnum = (get-index x index)
			    with y-dif fixnum = (get-index y index)
			    with o-dif fixnum = (get-index out index)
			    for i fixnum upfrom 0 below (the fixnum (!shape x index))
			    do (next
				   (1+ index)
				   (+ first-index-x (the fixnum (* i x-dif)))
				   (+ first-index-y (the fixnum (* i y-dif)))
				   (+ first-index-o (the fixnum (* i o-dif))))))
		     ((null bx)
		      (loop with x-dif fixnum = (get-index x index)
			    ;with y-dif fixnum = (get-index y index)
			    with o-dif fixnum = (get-index out index)
			    for i fixnum upfrom 0 below by
			    do (next
				(1+ index)
				(+ first-index-x (the fixnum (* i x-dif)))
				first-index-y
				(+ first-index-o (the fixnum (* i o-dif))))))
		     ((null by)
		      (loop ;with x-dif fixnum = (get-index x index)
			    with y-dif fixnum = (get-index y index)
			    with o-dif fixnum = (get-index out index)
			    for i fixnum upfrom 0 below bx
			    do (next
				(1+ index)
				first-index-x
				(+ first-index-y (the fixnum (* i y-dif)))
				(+ first-index-o (the fixnum (* i o-dif))))))
		     (T nil)))
		 nil))
	(next 0 0 0 0))
      (data out))))

(defun fill-with-d1 (shape i)
  (let ((index -1))
    (map 'list (lambda (x)
		 (declare (ignore x))
		 (incf index 1)
		 (cond
		   ((= i index)
		    1)
		   (T 0)))
	 shape)))

(defun broadcasting-apply-mgl (function x y &optional (broadcasts nil))					; still node debugged but it used mgl-mat's APIs
  (declare (optimize (speed 3) (safety 0))
	   (type symbol function)
	   (type waffetensor x y))
					; assume that (!dims x) == (!dims y)
  (unless (= (!dims x) (!dims y))
    (error "KernelError: Can't broadcasting ~a and ~a" x y))

  (let* ((dims (or broadcasts (cl-waffe::broadcasting x y)))
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
	       (let ((subscripts (fill-with-d1 shape dim)))
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
			     (dotimes (i (the fixnum (nth dim-currently-processing result-shape)))
			       (declare (type fixnum i))
			       (explore-batch (cdr dims-x)
					      (cdr dims-y)
					      (cdr dims-o)
					      (x-step-index x-index i repeat-instruction-x dim-currently-processing)
					      (y-step-index y-index i repeat-instruction-y dim-currently-processing)
					      (o-step-index o-index i dim-currently-processing)
					      (1+ dim-currently-processing)))
					; Todo pdotimes
			     (dotimes (i (the fixnum (nth dim-currently-processing result-shape)))
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
				     (axpy! -1.0 (data y) (data out))
				     (.+! scal (data out)))
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
				  (mgl-mat:with-ones (tmp tmp-size)
				    (fill! 1.0 tmp)
				    (case function
				      (:+
				       (fill! 1.0 (data out))
				       (scale-columns! (data row-x) tmp)
				       (scale-rows! (data columns-y) (data out))
				       (axpy! 1.0 tmp (data out)))
				      (:-
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
	  (data out))))))


