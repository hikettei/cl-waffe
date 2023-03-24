
(in-package :cl-waffe.backends.mgl)

(deftype batch-size ()
  `(integer 0 100000000))

(defun %matmul ())

; %pmatmul <- with-thread-cached-matで並列化

(declaim (ftype (function (boolean waffetensor waffetensor waffetensor &optional (or null mat) boolean boolean) mgl-mat:mat) matmul-tensor))
; Note: matmul would return unexcepted value if x or y is displaced.
; To prevent this, we probably need to create copy in advance.
(defun matmul-tensor (enable-optimize? o x y
		      &optional (output-to nil) (trans-a? nil) (trans-b? nil))
  (declare (optimize (speed 3))
	   (ignore o)
	   (type boolean enable-optimize?)
	   (type waffetensor o x y))
  (let* ((transpose-map `(,(or trans-a? (is-transpose? x))
			  ,(or trans-b? (is-transpose? y))))
	 (x1 (value x :ignore-transpose t))
	 (y1 (value y :ignore-transpose t)))
    (declare (type mat x1 y1))

    (unless (or (<= (length (the list (mat-dimensions x1))) 3)
		(<= (length (the list (mat-dimensions y1))) 3))
      (error "cl-waffe.backends.mgl:matmul-tensor Matmul only supports following: 2d * 2d, 2d * 3d, 3d * 2d, 3d * 3d."))

    ; x and y is a mat which isn't trasposed even when transpose is called befrore.
    ; transpose-mat indicates x or y called trasnposed before calling matmul. 
    (let ((x-dims (the list (mat-dimensions x1)))
	  (y-dims (the list (mat-dimensions y1))))
      (cond
	((and (= (length x-dims) 2)
	      (= (length y-dims) 2))

	 ; Todo: check shapes

	 (let ((out-dim `(,(if (car transpose-map)
			       (car (reverse (mat-dimensions x1)))
			       (car (mat-dimensions x1)))
			  ,(if (second transpose-map)
			       (second (reverse (mat-dimensions y1)))
			       (second (mat-dimensions y1))))))
	   (let ((out (or output-to (make-mat out-dim))))
	     (matmul-tensor-2d
	      out
	      x1
	      y1
	      (car transpose-map)
	      (second transpose-map))
	     out)))
	((and (= (length x-dims) 3)
	      (= (length y-dims) 2))
	 (let ((out-dim `(,(car x-dims)
			  ,(if (car transpose-map)
			       (nth 2 x-dims)
			       (nth 1 x-dims))
			  ,(if (second transpose-map)
			       (nth 0 y-dims)
			       (nth 1 y-dims))))
	       (displace-first (mat-displacement x1))
	       (shape-first    (mat-dimensions x1)))
	   (let ((out (or output-to (make-mat out-dim))))
	     (dotimes (i (the fixnum (car x-dims)))
	       (declare (type batch-size i)) ; note here!!!
	       (reshape-and-displace! out
				      (cdr out-dim)
				      (the fixnum (* i
						     (the batch-size (nth 1 out-dim))
						     (the batch-size (nth 2 out-dim)))))
	       
	       (reshape-and-displace!
		x1
		(cdr shape-first)
		(the fixnum (* i
			       (the batch-size (nth 1 shape-first))
			       (the batch-size (nth 2 shape-first)))))
	       (matmul-tensor-2d out x1 y1 (car transpose-map) (second transpose-map)))
	     (reshape-and-displace! out out-dim 0)
	     (reshape-and-displace! x1 shape-first displace-first)
	     out)))
	((and (= (length x-dims) 2)
	      (= (length y-dims) 3))
	 (let ((out-dim `(,(car y-dims)
			  ,(if (car transpose-map)
			       (nth 1 x-dims)
			       (nth 0 x-dims))
			  ,(if (second transpose-map)
			       (nth 1 y-dims)
			       (nth 2 y-dims))))
	       (displace-first (mat-displacement y1))
	       (shape-first    (mat-dimensions y1)))
	   (let ((out (or output-to (make-mat out-dim))))
	     (dotimes (i (the fixnum (car y-dims)))
	       (declare (type batch-size i))
	       (reshape-and-displace! out
				      (cdr out-dim)
				      (the fixnum
					   (* i
					      (the batch-size
						   (nth 1 out-dim))
					      (the batch-size
						   (nth 2 out-dim)))))
	       (reshape-and-displace! y1
				      (cdr shape-first)
				      (the fixnum
					   (* i
					      (the batch-size
						   (nth 1 shape-first))
					      (the batch-size
						   (nth 2 shape-first)))))
	       (matmul-tensor-2d out x1 y1
				 (car transpose-map)
				 (second transpose-map)))
	     (reshape-and-displace! out out-dim 0)
	     (reshape-and-displace! y1 shape-first displace-first)
	     out)))
	((= (length x-dims) (length y-dims))
	 ; Otherwise, Batch Filter is adapted

	 (let* ((dims (1- (length x-dims)))
		(batch-dims
		  (loop for i fixnum upfrom 0 below (- dims 1)
			unless (= (the fixnum (nth i x-dims))
				  (the fixnum (nth i y-dims)))
			  do (error "cl-waffe.backends.mgl:matmul-tensor: Operands could not broadcasted together. ~a and ~a. ~ath dims should be satisfy: ~a = ~a" x-dims y-dims i (nth i x-dims) (nth i y-dims))
			collect (nth i x-dims)))
		(output-tmp-dim ; (... 3 5) @ (... 5 3) -> (3 3)
		  `(,(if (car transpose-map)
			 (nth dims x-dims)
			 (nth (1- dims) x-dims))
		    ,(if (second transpose-map)
			 (nth (1- dims) y-dims)
			 (nth dims y-dims))))
		(out (or output-to (make-mat `(,@batch-dims ,@output-tmp-dim))))
		(displace-first1 (mat-displacement x1))
		(displace-first2 (mat-displacement y1))
		(out-dim (mat-dimensions out))
		(shape-first1 (mat-dimensions x1))
		(shape-first2 (mat-dimensions y1)))
	   (dotimes (i (the fixnum (car batch-dims)))
	     (declare (type batch-size i))
	     (reshape-and-displace! ;e.g.: (n 3 5) => k + (3 5) + m, k+m=n
	      out
	      (cdr out-dim)
	      (the fixnum
		   (* i
		      (the batch-size
			   (nth 1 out-dim))
	              (the batch-size
			   (nth 2 out-dim)))))
	     
	     (reshape-and-displace! x1
				    (cdr shape-first1)
				    (the fixnum
					 (* i
					    (the batch-size
						 (nth 1 shape-first1))
					    (the batch-size
						 (nth 2 shape-first1)))))
	     
	     (reshape-and-displace! y1
				    (cdr shape-first2)
				    (the fixnum
					 (* i
					    (the batch-size
						 (nth 1 shape-first2))
					    (the batch-size
						 (nth 2 shape-first2)))))

	     ; displace tensors (i.e: make it 2d and 2d) and apply matmul.

	     (matmul-tensor enable-optimize?
			    x
			    (const x1)
			    (const y1)
			    out
			    (car transpose-map)
			    (second transpose-map))
	     ; reset displace
	     (reshape-and-displace! x1 shape-first1 displace-first1)
	     (reshape-and-displace! y1 shape-first2 displace-first2))

	   (reshape-and-displace!
	    out
	    out-dim
	    0)
	   out))
	(T (error "cl-waffe.backends.mgl:matmul-tensor Operands could not broadcasted together. Can't multiply ~a and ~a. These tensors' dims must be <= 3" x y))))))

(declaim (ftype
	  (function
	   (mat mat mat boolean boolean)
	   mat)
	  matmul-tensor-2d))
(defun matmul-tensor-2d (out x y ta? tb?)
  (declare (optimize (speed 3) (safety 0)))
  (gemm! 1.0 x y 0.0 out :transpose-a? ta? :transpose-b? tb?))
