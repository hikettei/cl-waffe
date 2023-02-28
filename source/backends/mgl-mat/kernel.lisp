
(in-package :cl-waffe.backends.mgl)

; Todo Rewrite with define-lisp-kernel

(defmacro will-be-destructed (tensor)
  `(waffetensor-thread-data ,tensor))

(defmacro wanted-to-be-destructed? (tensor)
  (cl-waffe::waffetensor-is-next-destruct? tensor))

(defun create-thread-idx (thread-info &optional (ident ""))
  "Thread format: <Thread_IDx>+<Count_N>"
  (if thread-info
      (intern (format nil "~a+~a~a"
		      ident
		      (cl-waffe::waffenodethread-thread-idx thread-info)
		      (cl-waffe::waffenodethread-cache-n thread-info))
	      :keyword)
      (gensym)))

(defun decide-out-buffer (out args enable-optim copy?)
  (declare (optimize (speed 3))
	   (inline decide-out-buffer2
		   decide-out-buffer3))
  (let ((args (if (typep args 'waffetensor)
		  args
		  (sysconst args))))
    (value args)
    (if (null out)
	(decide-out-buffer3 out (data args) enable-optim copy?)
	(typecase args
	  (waffetensor
	   (decide-out-buffer2 out (data args) enable-optim copy?))))))
     
(defun decide-out-buffer2 (out
			   args
			   enable-optim
			   copy?)
  (declare (optimize (speed 3) (space 0)))
  (if (not (null (waffetensor-thread-data out)))
      (let* ((thread-info (waffetensor-thread-data out))
	     (idx (create-thread-idx thread-info)))
	(value out)
	(with-cache (result out :place idx :copy copy?)
	  result))
      (decide-out-buffer nil args enable-optim copy?)))

(defun decide-out-buffer3 (out
			   args
			   enable-optim
			   copy?)
  (declare (optimize (speed 3) (space 0) (safety 0))
	   (ignore out))
  (if enable-optim
      args
      (if copy?
	  (copy-mat args)
	  (make-mat (mat-dimensions args)))))

(declaim (ftype (function (mgl-mat:mat fixnum &key (:axis fixnum)) mgl-mat:mat) mgl-repeat))
(defun mgl-repeat (tensor n &key axis)
  (declare (optimize (speed 3) (space 0) (safety 0) (debug 0))
	   (type mgl-mat:mat tensor)
	   (type fixnum n axis))
  (if (>= axis 0)
      (if (>= (length (the list
			   (mat-dimensions tensor)))
	      2)
	  (mgl-mat:stack
	   axis
	   (loop for i below n collect tensor))
					; when dims=1
	  (mgl-repeat (mgl-mat:reshape
		       tensor
		       `(,@(mgl-mat:mat-dimensions tensor) 1))
 		      n :axis axis))
      (error "axis=-1")))

					;(declaim (ftype (function (mgl-mat:mat waffesupporteddatatype) mgl-mat:mat) trasposedmgl-full-like mgl-full-like))
(defun mgl-full-like (tensor value)
  (declare (optimize (speed 3) (safety 0) (debug 0))
	   (type mat tensor))
  (format t "Warning: making new tensor~%")
  (make-mat (mat-dimensions tensor)
	    :initial-element value))

(defun transposed-mgl-full-like (tensor value)
  (declare (optimize (speed 3) (safety 0) (debug 0))
	   (type mat tensor))
  (format t "Warning: making new tensor~%")
  (let ((dims (mat-dimensions tensor)))
    (declare (type cons dims))
    (make-mat (reverse dims) :initial-element value)))

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
					; Todo: CUDA Support
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
			    with y-dif fixnum = (get-index y index)
			    with o-dif fixnum = (get-index out index)
			    for i fixnum upfrom 0 below by
			    do (next
				(1+ index)
				(+ first-index-x (the fixnum (* i x-dif)))
				first-index-y
				(+ first-index-o (the fixnum (* i o-dif))))))
		     ((null by)
		      (loop with x-dif fixnum = (get-index x index)
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

(defparameter *v2v-operations* `(:add :sub :mul :div :dot :matmul))
(defparameter *abort-delay-instruction* :matmul)

(defun lazy-eval-transpose (tensor args)
  "Lazy eval's format is following: (free-args shape? return-calculated-value?)"
  (declare (ignore args))
  (if (typep tensor 'waffetensor)
      (error "KernelError: lazy-eval-transpose -> tensor must not be waffetensor."))
  (labels ((LazyTranspose (given-tensor
			   return-shape?
			   compile-and-step?
			   &optional
			     ignore?
			     return-node-info)
	     (declare (ignore given-tensor))
					; given-tensor is always lazytranspsoed.
	     (cond
	       (ignore?
		nil)
	       (return-shape?
					; Return transposed dims (for 2d only) for 3d is todo.
		(let ((result (!shape (sysconst tensor))))
		  (declare (type list result))
		  (case (length result)
		    (1 (reverse result))
		    (2 (reverse result))
		    (T `(,@(subseq result 0 (- (length result) 2))
			 ,@(reverse (subseq result (- (length result) 2)
					    (length result))))))))
	       (return-node-info
		(values :lazy-transpose nil nil nil))
	       (compile-and-step?
					; Transpose is evaluated (its slow)
		(transpose (compile-and-run-lazy (sysconst tensor))))
	       (T
					; The Last Transpose is skipped, returning untransposed tensor
					; this block will be called by (value ~ :ignore-transpose t)
					; so, if tensor should function, this will be evaluated.
		(value (sysconst tensor))))))
    #'LazyTranspose))

(defmacro deliv-delay (tensor func &rest args)
  (declare (ignore func))
  (lazy-eval-transpose tensor args))

(defun next-delay (delay state)
  (if (typep delay 'function)
      (funcall delay nil nil state)
      delay))

(defmacro abort-delay (delay)
  `(next-delay ,delay nil))

(defmacro receive-delay (delay)
  `(next-delay ,delay t))

(defun ensure-shape (ope x)
  (declare (type keyword ope))
  (if (find ope *v2v-operations*)
      (if (or (typep x 'mat) (typep x 'function))
	  (if (eq ope *abort-delay-instruction*)
	      (abort-delay x)
	      (receive-delay x))
	  (if (eq ope :matmul)
	      (transposed-mgl-full-like x 0)
	      (mgl-full-like x 0)))))

(defun infomation ())

(declaim (ftype (function (boolean waffetensor waffetensor) mgl-mat:mat)
		inv-tensor
		sqrt-tensor
		log-tensor
		tanh-tensor
		exp-tensor))

					; kernel function must be following:
					; fname (enable-optimize? args-tensor1 args-tensor2 ... mat1 mat2 ....)
					; Otherwise ignore jit can't return correctly.
					; Todo: Write macro in order to define this.

(defmacro define-waffe-kernel (name
			       args
			       args-mat
			       &key
				 (ignore-optimize nil)
				 (jit nil)
				 (mat-mat nil)
				 (scal-mat nil)
				 (mat-scal nil))
  "(define-waffe-kernel-function add-tensor (a b) (x y)
      :jit '+)
      :mat-mat ~
      :scal-mat ~
      :mat-scal ~"
  `(defun ,name (enable-optimize? ,@args &key (output nil) (overwrite nil))
     ,(unless ignore-optimize
	`(declare (optimize (speed 3) (space 0))
		  (type boolean enable-optimize? overwrite)
		  (type waffetensor ,@args)))
     ,(unless (null jit)
	 ; place jit trigger.
	`(return-and-lazy-eval
	  ,name
	  ',jit
	  ,(car args)
	  (list ,@(cdr args))))
     
     ; if jit triggered, the form below never called.

     (macrolet ((get-out-buffer (tensor &key (copy nil))
		  `(cond
		     (overwrite (value ,tensor))
		     (output
		      (if ,copy
			  (copy! (value ,tensor) output))
		      output)
		     ((cl-waffe::waffetensor-is-next-destruct? ,tensor)
		      (value ,tensor))
		     (T (decide-out-buffer ,tensor (value ,tensor) enable-optimize? ,copy)))))
       
       (let (,@(map 'list (lambda (target val)
			    `(,target (value ,val)))
		    args-mat args))
	 (cond
	   ((or (= (length ',args) 1)
		(and (= (length ',args) 2)
		     (and (typep ,(car args-mat) 'mat)
			  (typep ,(second args-mat) 'mat)))
		(and (>= (length ',args) 3)))
	    ,@mat-mat)
	   ((or (= (length ',args) 1)
		(and (= (length ',args) 2)
		     (and (not (typep ,(car args-mat) 'mat))
			  (typep ,(second args-mat) 'mat)))
		(and (>= (length ',args) 3)))
	    ,(if (null scal-mat)
		 `(progn ,@mat-mat)
		 `(progn ,@scal-mat)))
	   ((or (= (length ',args) 1)
		(and (= (length ',args) 2)
		     (and (typep ,(car args-mat) 'mat)
			  (not (typep ,(second args-mat) 'mat))))
		(and (>= (length ',args) 3)))
	    ,(if (null mat-scal)
		 `(progn ,@mat-mat)
		 `(progn ,@mat-scal)))
	   (T (error "define-waffe-kernel: arguments didn't hit.")))))))

(define-waffe-kernel kernel-add (x y) (x1 y1)
  :jit +
  :mat-scal ((let ((o (get-out-buffer x :copy t)))
	       (.+! y1 o)))
  :scal-mat ((let ((o (get-out-buffer y :copy t)))
	       (.+! x1 o)))
  :mat-mat ((if (equal (!shape x) (!shape y))
		(cond
		  ((will-be-destructed x)
		   (let ((o (get-out-buffer x :copy t)))
		     (axpy! 1.0 y1 o)))
		  ;((will-be-destructed y)
		   ;(axpy! 1.0 x1 (get-out-buffer y :copy t)))
		  (T (axpy! 1.0 y1 (get-out-buffer x :copy t))))
		(broadcasting-apply :+ x y))))

(define-waffe-kernel kernel-sub (x y) (x1 y1)
  :jit -
  :mat-scal ((let ((o (get-out-buffer x :copy t)))
	       (.+! (the single-float (* -1.0 (the single-float y1)))
		    o)))
  :scal-mat ((let ((o (get-out-buffer y :copy t)))
	       (.+! x1
		    (scal! -1.0 o))))
  :mat-mat ((if (equal (!shape x) (!shape y))
		(cond
		  ((will-be-destructed x)
		   (let ((o (get-out-buffer x :copy t)))
		     (axpy! -1.0 y1 o)))
		  ;((will-be-destructed y)
		  ; (axpy! 1.0 x1 (scal! -1.0 (get-out-buffer y :copy t))))
		  (T (axpy! -1.0 y1 (get-out-buffer x :copy t))))
		(broadcasting-apply :- x y))))

(define-waffe-kernel kernel-mul (x y) (x1 y1)
  :jit *
  :mat-scal ((let ((o (get-out-buffer x :copy t)))
	       (scal! y1 o)))
  :scal-mat ((let ((o (get-out-buffer y :copy t)))
	       (scal! x1 o)))
  :mat-mat ((if (equal (!shape x) (!shape y))
		(cond
		  ((will-be-destructed x)
		   (let ((o (get-out-buffer x :copy nil)))
		     (geem! 1.0 x1 y1 0.0 o)))
		  ;((will-be-destructed y)
		  ; (geem! 1.0 x1 y1 0.0 (get-out-buffer y :copy nil)))
		  (T (geem! 1.0 x1 y1 0.0 (get-out-buffer x :copy nil))))
		(broadcasting-apply :* x y))))

(define-waffe-kernel kernel-inv (x) (x1)
  :jit /
  :mat-mat ((.inv! (get-out-buffer x :copy t))))

(define-waffe-kernel kernel-div (x y) (x1 y1)
  :jit /
  :mat-scal ((let ((o (get-out-buffer x :copy t)))
	       (scal! (/ y1) o)))
  :scal-mat ((unless (= x1 1)
	       (error "cl-waffe.backends.mgl:kernel-inv excepts x1 to be 1"))
	     (let ((o (get-out-buffer y :copy t)))
	       (.inv! o))))

(defun dot-tensor (enable-optimize? out x y)
  (declare (ignore enable-optimize? out))
  (mgl-mat:dot (value x) (value y)))

(defun is-transpose? (tensor)
  (declare (type waffetensor tensor))
  (typecase (data tensor)
    (function
     (multiple-value-bind
	   (node-type)
	 (funcall (data tensor) nil nil nil nil t)
       (eql node-type :lazy-transpose)))
    (T nil)))

(declaim (ftype (function (boolean waffetensor waffetensor waffetensor) mgl-mat:mat)
		pow-tensor
		compare-tensor
		sum-tensor))

(declaim (ftype (function (boolean waffetensor waffetensor waffetensor &optional (or null mat) boolean boolean) mgl-mat:mat) matmul-tensor))
; Note: matmul would return unexcepted value if x or y is displaced.
; To prevent this, we probably need to create copy in advance.
(defun matmul-tensor (enable-optimize? o x y &optional (output-to nil) (trans-a? nil) (trans-b? nil))
  (declare (optimize (speed 3) (space 0) (safety 1))
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

    (warranty x)
    (warranty y)

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
	       (declare (type (integer 0 10000000) i)) ; note here!!!
	       (reshape-and-displace! out
				      (cdr out-dim)
				      (the fixnum (* i
						     (the (integer 0 10000000) (nth 1 out-dim))
						     (the (integer 0 10000000) (nth 2 out-dim)))))
	       
	       (reshape-and-displace!
		x1
		(cdr shape-first)
		(the fixnum (* i
			       (the (integer 0 10000000) (nth 1 shape-first))
			       (the (integer 0 10000000) (nth 2 shape-first)))))
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
	       (declare (type (integer 0 10000000) i)) ; note here!!!
	       (reshape-and-displace! out
				      (cdr out-dim)
				      (the fixnum
					   (* i
					      (the (integer 0 10000000)
						   (nth 1 out-dim))
					      (the (integer 0 10000000)
						   (nth 2 out-dim)))))
	       (reshape-and-displace! y1
				      (cdr shape-first)
				      (the fixnum
					   (* i
					      (the (integer 0 10000000)
						   (nth 1 shape-first))
					      (the (integer 0 10000000)
						   (nth 2 shape-first)))))
	       (matmul-tensor-2d out x1 y1
				 (car transpose-map)
				 (second transpose-map)))
	     (reshape-and-displace! out out-dim 0)
	     (reshape-and-displace! y1 shape-first displace-first)
	     out)))
	((= (length x-dims) (length y-dims))
	 ; Otherwise, Batch Filter is adapted

	 (when (>= (length x-dims) 4)
	   (format t "Warning: currently cl-waffe doesn't support matmul for >= 3d. It possibly has performance problems."))

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
		(out-tmp (make-mat output-tmp-dim))
		(displace-first1 (mat-displacement x1))
		(displace-first2 (mat-displacement y1))
		(out-dim (mat-dimensions out))
		(shape-first1 (mat-dimensions x1))
		(shape-first2 (mat-dimensions y1)))
	   (dotimes (i (the fixnum (car batch-dims)))
	     (declare (type (integer 0 10000000) i))
	     (reshape-and-displace! ;e.g.: (n 3 5) => k + (3 5) + m, k+m=n
	      out
	      (cdr out-dim)
	      (the fixnum
		   (* i
		      (the (integer 0 10000000)
			   (nth 1 out-dim))
	              (the (integer 0 10000000)
			   (nth 2 out-dim)))))
	     
	     (reshape-and-displace! x1
				    (cdr shape-first1)
				    (the fixnum
					 (* i
					    (the (integer 0 10000000)
						 (nth 1 shape-first1))
					    (the (integer 0 10000000)
						 (nth 2 shape-first1)))))
	     
	     (reshape-and-displace! y1
				    (cdr shape-first2)
				    (the fixnum
					 (* i
					    (the (integer 0 10000000)
						 (nth 1 shape-first2))
					    (the (integer 0 10000000)
						 (nth 2 shape-first2)))))

	     ; displace tensors (i.e: make it 2d and 2d) and apply matmul.

	     (matmul-tensor enable-optimize?
			    x
			    (const x1)
			    (const y1)
			    out-tmp
			    (car transpose-map)
			    (second transpose-map))
	     ; reset displace
	     (reshape-and-displace! x1 shape-first1 displace-first1)
	     (reshape-and-displace! y1 shape-first2 displace-first2)

	     (copy! out-tmp out))

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
  (declare (optimize (speed 3) (space 0) (safety 1)))
  (gemm! 1.0 x y 0.0 out :transpose-a? ta? :transpose-b? tb?)
  out)

(define-waffe-kernel kernel-log (x) (x1)
  :jit log
  :mat-mat ((.log! (get-out-buffer x :copy t))))

(define-waffe-kernel kernel-exp (x) (x1)
  :jit exp
  :mat-mat ((.exp! (get-out-buffer x :copy t))))

(define-waffe-kernel kernel-sqrt (x) (x1)
  :jit sqrt
  :mat-mat ((.sqrt! (get-out-buffer x :copy t))))

(define-waffe-kernel kernel-pow (x n) (x1 n1)
  :jit pow
  :mat-mat ((.expt! (get-out-buffer x :copy t) n1)))

(define-waffe-kernel kernel-sin (x) (x1)
  :jit sin
  :mat-mat ((.sin! (get-out-buffer x :copy t))))

(define-waffe-kernel kernel-cos (x) (x1)
  :jit cos
  :mat-mat ((.cos! (get-out-buffer x :copy t))))

(define-waffe-kernel kernel-tan (x) (x1)
  :jit tan
  :mat-mat ((.tan! (get-out-buffer x :copy t))))


(define-waffe-kernel kernel-sinh (x) (x1)
  :jit sinh
  :mat-mat ((.sinh! (get-out-buffer x :copy t))))

(define-waffe-kernel kernel-cosh (x) (x1)
  :jit cosh
  :mat-mat ((.cosh! (get-out-buffer x :copy t))))

(define-waffe-kernel kernel-tanh (x) (x1)
  :jit tanh
  :mat-mat ((.tanh! (get-out-buffer x :copy t))))


(define-waffe-kernel kernel-asin (x) (x1)
  :ignore-optimize t
  :jit asin
  :mat-mat ((with-facet (r ((get-out-buffer x :copy t) 'backing-array))
	      (declare (type (simple-array single-float) r))
	      (loop for i fixnum upfrom 0 below (!size x)
		    do (setf (aref r i) (asin (aref r i))))
	      r)))

(define-waffe-kernel kernel-acos (x) (x1)
  :ignore-optimize t
  :jit acos
  :mat-mat ((with-facet (r ((get-out-buffer x :copy t) 'backing-array))
	      (declare (type (simple-array single-float) r))
	      (loop for i fixnum upfrom 0 below (!size x)
		    do (setf (aref r i) (acos (aref r i))))
	      r)))

(define-waffe-kernel kernel-atan (x) (x1)
  :ignore-optimize t
  :jit atan
  :mat-mat ((with-facet (r ((get-out-buffer x :copy t) 'backing-array))
	      (declare (type (simple-array single-float) r))
	      (loop for i fixnum upfrom 0 below (!size x)
		    do (setf (aref r i) (atan (aref r i))))
	      r)))

(define-waffe-kernel kernel-asinh (x) (x1)
  :ignore-optimize t
  :jit asinh
  :mat-mat ((with-facet (r ((get-out-buffer x :copy t) 'backing-array))
	      (declare (type (simple-array single-float) r))
	      (loop for i fixnum upfrom 0 below (!size x)
		    do (setf (aref r i) (asinh (aref r i))))
	      r)))

(define-waffe-kernel kernel-acosh (x) (x1)
  :ignore-optimize t
  :jit acosh
  :mat-mat ((with-facet (r ((get-out-buffer x :copy t) 'backing-array))
	      (declare (type (simple-array single-float) r))
	      (loop for i fixnum upfrom 0 below (!size x)
		    do (setf (aref r i) (acosh (aref r i))))
	      r)))

(define-waffe-kernel kernel-atanh (x) (x1)
  :ignore-optimize t
  :jit atanh
  :mat-mat ((with-facet (r ((get-out-buffer x :copy t) 'backing-array))
	      (declare (type (simple-array single-float) r))
	      (loop for i fixnum upfrom 0 below (!size x)
		    do (setf (aref r i) (atanh (aref r i))))
	      r)))

(defun compare-tensor (enable-optim out x y)
  (declare (optimize (speed 3) (space 0) (safety 1))
	   (type boolean enable-optim)
           (type waffetensor out x y))
					; Todo do lazy
  (let ((o (decide-out-buffer out x enable-optim t)))
    (mgl-mat:.<! (value y) o)))

(defun sum-tensor (is-first-time-call? out x y)
  (declare (optimize (speed 3) (space 0) (safety 0))
           (type boolean is-first-time-call?)
           (type waffetensor out x y)
	   (ignore is-first-time-call? out))
  ; Todo Optimize 

  (warranty x)
  (warranty y)
  
  (value x)
  (value y)
  
  (let* ((dims (mat-dimensions (data x)))
	 (dims (if (and (= 1 (the fixnum (car (last dims))))
			(= 3 (length dims)))
		   (butlast dims)
		   dims))
	 (x1 (value x))
	 (dims (case (data y)
		 (1 `(,@(list (car dims)) 1))
		 (0 `(1 ,@(cdr dims)))
		 (T (error "Sum only supports a 2d matrix")))))
    (let ((o (mgl-mat:make-mat dims :initial-element 0.0)))
      (mgl-mat:sum! x1 o :axis (data y) :beta 0.0)
      (mgl-mat:reshape! o dims)
      (if (equal dims `(1 1))
	  (mat-as-scalar o)
	  o))))

(defun mean-tensor (is-first-time-call? out x y)
  (declare (optimize (speed 3) (space 0) (safety 1))
           (type boolean is-first-time-call?)
           (type waffetensor out x y)
	   (ignore is-first-time-call? out))

  (warranty x)
  (warranty y)

  (value x)
  (value y)

  (let* ((dims (mgl-mat:mat-dimensions (value x)))
	 (dims (if (and (= 1 (the fixnum (car (last dims))))
			(= 3 (length dims)))
		   (butlast dims)
		   dims))
	 (x1 (mgl-mat:reshape! (mgl-mat:copy-mat (value x)) dims))
	 (dims (case (data y)
		 (1 `(,@(list (car dims)) 1))
		 (0 `(1 ,@(cdr dims)))
		 (T (error "Sum only supports a 2d matrix")))))
    (let ((o (mgl-mat:make-mat dims :initial-element 0))
	  (s (nth (value y) (mgl-mat:mat-dimensions (value x)))))
      (mgl-mat:sum! x1 o :axis (data y) :beta 1)
      (mgl-mat:scal! (/ 1 (the integer s)) x1)
      (mgl-mat:reshape! o dims)
      (if (equal dims `(1 1))
	  (mgl-mat:mref o 0 0)
	  o))))

;(declaim (ftype (function (boolean waffetensor waffetensor waffetensor) mgl-mat:mat) reshape-tensor))
(defun reshape-tensor (enable-optimize out x y &key (output nil) (overwrite nil))
  (declare (optimize (speed 3) (space 0) (safety 1))
	   (type boolean enable-optimize)
	   (type waffetensor out x y))
  (let ((x1 (cond
	      ((not (null output))
	       output)
	      (overwrite
	       (value x))
	      ((cl-waffe::waffetensor-is-next-destruct? x)
	       (value x))
	      (T
	       (decide-out-buffer out (value x) enable-optimize t)))))
    (mgl-mat:reshape! x1 (data y))
    x1))

(define-lisp-kernel (bernoulli-lisp)
    ((mask :mat :io)
     (start-mask fixnum)
     (n fixnum)
     (p single-float))
  (loop for mi fixnum upfrom start-mask below (the fixnum (+ start-mask n))
	do (cond ((< (aref mask mi) p)
		  (setf (aref mask mi) 0.0))
		 (t
		  (setf (aref mask mi) 1.0)))))

					; having not cuda gpus, i can't test this code lol T_T
					;(define-cuda-kernel (bernoulli-cuda)
					;    (void ((mask :mat :io) (x :mat :io) (n int) (p float)))
					;  (let ((i (+ (* block-dim-mask block-idx-mask) thread-idx-mask)))
					;    (when (< i n)
					;      (if (< (aref x i) p)
					;	  (set (aref x i) 0.0)
					;          (set (aref x i) 1.0)))))

(defun bernoulli-tensor (enable-optimize out return-tensor rate)
  (declare (type boolean enable-optimize)
	   (type waffetensor return-tensor rate))
  (let ((o (decide-out-buffer out return-tensor enable-optimize nil)))
    (mgl-mat:uniform-random! o)
    (if (use-cuda-p (data return-tensor))
	(progn
	  (error "having not gpus, i've not tested cl-waffe.backends.mgl:bernoulli-tensor yet in cuda.")
					;(bernoulli-cuda o (mat-size o) (data rate)
					;		:grid-dim (list (ceiling (mat-size o) 256) 1 1)
					;		:block-dim (list 256 1 1))
	  )
	(bernoulli-lisp o (mat-displacement o) (mat-size o) (data rate)))))


					;optimize is failed.
(define-lisp-kernel (embedding-forward-lisp)
    ((out :mat :output)
     (x :mat :input)
     (weights :mat :input)
     (n fixnum)
     (pad-idx fixnum)
     (embedding-dim fixnum))
  (loop for xi fixnum upfrom 0 below n
	do (cond
	     ((= pad-idx (round (aref x xi)))
	      nil)
	     (T
	      (loop for ei fixnum upfrom 0 below embedding-dim
		    do (setf (aref out (the fixnum
					    (+
					     (the fixnum
						  (* embedding-dim xi))
					     ei)))
			     (aref weights
				   (the fixnum
					(+ ei
					   (the fixnum
						(* (the fixnum
							(round (aref x xi)))
						   embedding-dim)))))))))))

(defun embedding-forward (enable-optimize x weights pad-idx)
  "(with-searching-calc-node :embedding-forward x weights pad-idx) -> embeddings"
  (declare (type boolean enable-optimize)
	   (type waffetensor x weights pad-idx)
	   (ignore enable-optimize))
  (let* ((batch-size (mat-dimension (data x) 0))
	 (total-size (mat-dimension (data x) 1))
	 (out (mgl-mat:make-mat `(,batch-size ,total-size ,(!shape weights 1))
				:initial-element 0.0)))
					; there's no cuda ver...
    (embedding-forward-lisp
     out
     (data x)
     (data weights)
     (mat-size (data x))
     (data pad-idx)
     (!shape weights 1))
    out))

(define-lisp-kernel (embedding-backward-lisp)
    ((out :mat :io)
     (x :mat)
     (dy :mat)
     (n fixnum)
     (pad-idx fixnum)
     (embedding-dim fixnum))
  (loop for xi of-type fixnum upfrom 0 below n
	do (cond
	     ((= pad-idx (round (aref x xi)))
	      nil)
	     (T
	      (loop for ei of-type fixnum upfrom 0 below embedding-dim
		    do (setf (aref out
				   (+ ei
				      (the fixnum
					   (* embedding-dim
					      (round (aref x xi))))))
			     (+ (aref out
				      (+ ei
					 (the fixnum
					      (* embedding-dim
						 (round (aref x xi))))))
				(aref dy (+ ei
					    (the fixnum
						 (* xi
						    embedding-dim)))))))))))

(defun embedding-backward (enable-optimize x dy weights pad-idx)
  (declare (type boolean enable-optimize)
	   (type waffetensor dy weights)
	   (ignore enable-optimize))
  (let ((out (mgl-mat:make-mat (!shape weights))))
    (embedding-backward-lisp
     out
     (data x)
     (data dy)
     (mgl-mat:mat-size (data x))
     (data pad-idx)
     (!shape dy 2))
    out))
#|
(declaim (ftype
	  (function
	   (keyword
	    boolean
	    waffetensor
	    waffetensor
	    cons
	    &key
	    (output (or null waffetensor))
	    (overwrite boolean))
	   (or mgl-mat:mat cl-waffe:waffedatatype))
	  dispatch-kernel))|#
(defun dispatch-kernel (function is-first-time-call? destructable-tensor destructable-tensor1 args &key (output nil) (overwrite nil))
  (declare (optimize (speed 3) (space 0) (safety 0))
	   (type keyword function)
	   (type boolean is-first-time-call?)
	   (type waffetensor destructable-tensor)
	   (type cons args))
  (case function
    (:add     (kernel-add
	       is-first-time-call?
	       destructable-tensor
	       destructable-tensor1
	       :output output
	       :overwrite overwrite))
    (:sub     (kernel-sub
	       is-first-time-call?
	       destructable-tensor
	       destructable-tensor1
	       :output output
	       :overwrite overwrite))
    (:mul     (kernel-mul
	       is-first-time-call?
	       destructable-tensor
	       destructable-tensor1
	       :output output
	       :overwrite overwrite))
    (:div     (kernel-div
	       is-first-time-call?
	       destructable-tensor
	       destructable-tensor1
	       :output output
	       :overwrite overwrite))
    (:dot     (dot-tensor is-first-time-call? destructable-tensor (car args) (second args)))
    (:matmul  (matmul-tensor is-first-time-call? destructable-tensor (car args) (second args)))    
    (:log     (kernel-log is-first-time-call? destructable-tensor
			  :output output
			  :overwrite overwrite))
    (:exp     (kernel-exp is-first-time-call? destructable-tensor
			  :output output
			  :overwrite overwrite))
    (:sqrt    (kernel-sqrt is-first-time-call? destructable-tensor
			   :output output
			   :overwrite overwrite))
    (:sin     (kernel-sin is-first-time-call? destructable-tensor
			  :output output
			  :overwrite overwrite))
    (:cos     (kernel-cos is-first-time-call? destructable-tensor
			  :output output
			  :overwrite overwrite))
    (:tan     (kernel-tan is-first-time-call? destructable-tensor
			  :output output
			  :overwrite overwrite))
    (:asin    (kernel-asin is-first-time-call? destructable-tensor
			   :output output
			   :overwrite overwrite))
    (:acos    (kernel-acos is-first-time-call? destructable-tensor
			   :output output
			   :overwrite overwrite))
    (:atan    (kernel-atan is-first-time-call? destructable-tensor
			   :output output
			   :overwrite overwrite))
    
    (:sinh    (kernel-sinh is-first-time-call? destructable-tensor
			   :output output
			   :overwrite overwrite))
    (:cosh    (kernel-cosh is-first-time-call? destructable-tensor
			   :output output
			   :overwrite overwrite))
    (:tanh    (kernel-tanh is-first-time-call? destructable-tensor
			   :output output
			   :overwrite overwrite))
    
    (:asinh   (kernel-asinh is-first-time-call? destructable-tensor
			    :output output
			    :overwrite overwrite))
    (:acosh   (kernel-acosh is-first-time-call? destructable-tensor
			    :output output
			    :overwrite overwrite))
    (:atanh   (kernel-atanh is-first-time-call? destructable-tensor
			    :output output
			    :overwrite overwrite))
    
    (:pow     (kernel-pow
	       is-first-time-call?
	       destructable-tensor
	       destructable-tensor1
	       :output output
	       :overwrite overwrite))
    (:sum     (sum-tensor is-first-time-call? destructable-tensor (car args) (second args)))
    (:mean    (mean-tensor is-first-time-call? destructable-tensor (car args) (second args)))
    (:reshape (reshape-tensor is-first-time-call? destructable-tensor (car args) (second args) :output output
	       :overwrite overwrite))
    (:<       (compare-tensor is-first-time-call? destructable-tensor (car args) (second args)))
    (:repeat  (mgl-repeat (data (car args)) (data (third args)) :axis (data (second args))))
    (:bernoulli (bernoulli-tensor is-first-time-call? destructable-tensor (car args) (second args)))
    (:transpose (lazy-eval-transpose
		 (data (car args))
		 nil))
    (:embedding-forward (embedding-forward nil (car args)
					   (second args)
					   (third args)))
    (:embedding-backward (embedding-backward nil
					     (car args)
					     (second args)
					     (third args)
					     (fourth args)))
    (T (error "~a is not yet implemented" function))))

