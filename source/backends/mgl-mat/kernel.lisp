
(in-package :cl-waffe.backends.mgl)

; Todo Rewrite with define-lisp-kernel

(defmacro duplicate-tensor (mat)
  ;`(let ((out (copy-mat ,mat)))
     ; matをgcしたい
   ;  (mgl-cube:destroy-cube ,mat)
    ; out))

  mat)

(defmacro apply-destruct (out)
  `(progn ; tensor=out
     (setf (waffetensor-is-data-destructed? ,out) t))
  nil)

(defmacro assure-destructed? (out) ;for automatic !modify
  `(progn
     (if (waffetensor-is-data-destructed? ,out)
       (error "Kernel Error: Modifying tensor that is not allowed to destruct"))
     (the mgl-mat:mat (data ,out))))

(defmacro will-be-destructed (tensor)
  `(waffetensor-is-node-tensor ,tensor))

(defgeneric decide-out-buffer (out args enable-optim))

(defmethod decide-out-buffer ((out waffetensor) (args waffetensor) enable-optim)
  (declare (optimize (speed 3) (space 0) (safety 0)))
  (decide-out-buffer out (data args) enable-optim))

(defmethod decide-out-buffer ((out waffetensor) (args mgl-mat:mat) enable-optim)
  (if (waffetensor-is-node-tensor out)
      args
      (duplicate-tensor args)))

(defmethod decide-out-buffer ((out null) (args waffetensor) enable-optim)
  (declare (optimize (speed 3) (space 0) (safety 0)))
  (decide-out-buffer out (data args) enable-optim))

(defmethod decide-out-buffer ((out null) (args mgl-mat:mat) enable-optim)
  (declare (ignore enable-optim))
  (with-ones (o (mat-dimensions args))
    (copy! args (reshape o (mat-dimensions args)))))

(declaim (ftype (function (mgl-mat:mat fixnum &key (:axis fixnum)) mgl-mat:mat) mgl-repeat))
(defun mgl-repeat (tensor n &key axis)
  (declare (optimize (speed 3) (safety 0) (debug 0))
	   (type mgl-mat:mat tensor)
	   (type fixnum n axis))
  (if (>= axis 0)
      (if (>= (length (the list
			   (mat-dimensions tensor)))
	      2)
          (with-ones (o (loop for i
			      upfrom 0
			      below (the fixnum (length
						 (the list (mat-dimensions tensor))))
			      collect (cond
					((= i axis)
					 (the fixnum (* n (the fixnum
						      (mat-dimension tensor i)))))
					(T (mat-dimension tensor i)))))
	    (mgl-mat:stack!
	     axis
	     (loop for i below n collect tensor)
	     o))
	  ; when dims=1
	  (mgl-repeat (mgl-mat:reshape!
		       tensor
		       `(,@(mgl-mat:mat-dimensions tensor) 1))
 		       n :axis axis))
      (error "axis=-1")))

;(declaim (ftype (function (mgl-mat:mat waffesupporteddatatype) mgl-mat:mat) trasposedmgl-full-like mgl-full-like))
(defun mgl-full-like (tensor value)
  (declare (optimize (speed 3) (safety 0) (debug 0))
	   (type mat tensor))
  (make-mat (mat-dimensions tensor)
	    :initial-element value))

(defun transposed-mgl-full-like (tensor value)
  (declare (optimize (speed 3) (safety 0) (debug 0))
	   (type mat tensor))
  (let ((dims (mat-dimensions tensor)))
    (declare (type cons dims))
    (make-mat (reverse dims) :initial-element value)))

(defparameter *v2v-operations* `(:add :sub :mul :div :dot :matmul))
(defparameter *abort-delay-instruction* :matmul)

(defmacro deliv-delay (tensor func &rest args)
  `(lambda (shape? step?)
     (if shape?
	 (reverse (mgl-mat:mat-dimensions ,tensor))
	 (if step?
	     (funcall ,func ,tensor ,@args) ; receive before node
	     ,tensor)))) ; abort before node

(defun next-delay (delay state)
  (if (typep delay 'function)
      (funcall delay nil state)
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

(defgeneric add-tensor (enable-optimize? out out1 x y))

(defmethod add-tensor (enable-optimize? (out waffetensor) (out1 waffetensor) (x mgl-mat:mat) (y mgl-mat:mat))
  (declare (optimize (speed 3) (space 0) (safety 0)))
  (cond
    ((will-be-destructed out)
     (let ((o (decide-out-buffer out x enable-optimize?)))
       (mgl-mat:axpy! 1.0 y o)
       o))
    ((will-be-destructed out1)
     (add-tensor enable-optimize? out1 out y x))
    (T (let ((o (decide-out-buffer out x enable-optimize?)))
	 (mgl-mat:axpy! 1.0 y o)
	 o))))

(defmethod add-tensor (enable-optimize? (out waffetensor) (out1 waffetensor) (x mgl-mat:mat) y)
  (declare (optimize (speed 3) (space 0) (safety 0))
	   (ignore out1))
  (let ((o (decide-out-buffer out x enable-optimize?)))
    (the mgl-mat:mat (mgl-mat:.+! y o))))

(defmethod add-tensor (enable-optimize? (out waffetensor) (out1 waffetensor) x (y mgl-mat:mat))
  (declare (optimize (speed 3) (space 0) (safety 0))
	   (ignore out))
  (let ((o (decide-out-buffer out1 y enable-optimize?)))
    (the mgl-mat:mat (mgl-mat:.+! x o))))

(defgeneric sub-tensor (enable-optimize? out out1 x y))
(defmethod sub-tensor (enable-optimize? (out waffetensor) (out1 waffetensor) (x mgl-mat:mat) (y mgl-mat:mat))
  (declare (optimize (speed 3) (space 0) (safety 0)))
  (cond
    ((will-be-destructed out)
     (let ((o (decide-out-buffer out x enable-optimize?)))
       (mgl-mat:axpy! -1.0 y o)
       o))
    ((will-be-destructed out1) ;x-y, -(y-x) = x-y
     (mgl-mat:scal! -1.0 (sub-tensor enable-optimize? out1 out y x)))
    (T (let ((o (decide-out-buffer out x enable-optimize?)))
	 (mgl-mat:axpy! -1.0 y o)
	 o))))

(defmethod sub-tensor (enable-optimize? (out waffetensor) (out1 waffetensor) (x mgl-mat:mat) y)
  (declare (optimize (speed 3) (space 0) (safety 0))
	   (ignore out1))
  (let ((o (decide-out-buffer out x enable-optimize?)))
    (the mgl-mat:mat (mgl-mat:.+! (* -1.0 (the single-float y)) o))))

(defmethod sub-tensor (enable-optimize? (out waffetensor) (out1 waffetensor) x (y mgl-mat:mat))
  (declare (optimize (speed 3) (space 0) (safety 0))
	   (ignore out))
  (let ((o (decide-out-buffer out1 y enable-optimize?)))
    (the mgl-mat:mat (mgl-mat:.+! (* -1.0 (the single-float x)) o))))

(defgeneric mul-tensor (enable-optimize? out out1 x y))
(defmethod mul-tensor (enable-optimize? (out waffetensor) (out1 waffetensor) (x mgl-mat:mat) (y mgl-mat:mat))
  (declare (optimize (speed 3) (space 0) (safety 0)))

  (cond
    ((will-be-destructed out)
     (let ((o (decide-out-buffer out x enable-optimize?)))
       (mgl-mat:geem! 1 x y 0 o)))
    ((will-be-destructed out1)
     (mul-tensor enable-optimize? out1 out y x))
    (T
     (let ((o (mgl-mat:make-mat (mgl-mat:mat-dimensions x))))
       (mgl-mat:geem! 1 x y 0 o)))))

(defmethod mul-tensor (enable-optimize? (out waffetensor) (out1 waffetensor) x (y mgl-mat:mat))
  (declare (optimize (speed 3) (space 0) (safety 0))
	   (ignore out))
  (let ((o (decide-out-buffer out1 y enable-optimize?)))
	(mgl-mat:scal! x o)))

(defmethod mul-tensor (enable-optimize? (out waffetensor) (out1 waffetensor) (x mgl-mat:mat) y)
  (declare (optimize (speed 3) (space 0) (safety 0))
	   (ignore out1))
  (let ((o (decide-out-buffer out x enable-optimize?)))
	(mgl-mat:scal! y o)))

(defun inv-tensor (enable-optim out x)
  (declare (optimize (speed 3) (space 0) (safety 0))
	   (type boolean enable-optim)
           (type waffetensor out)
	   (type waffetensor x))
  (let ((o (decide-out-buffer out x enable-optim)))
    (mgl-mat:.inv! o)
    (the mgl-mat:mat o)))

(defgeneric div-tensor (enable-optimize? out out1 x y))

(defmethod div-tensor (enable-optimize? out out1 (x mgl-mat:mat) (y mgl-mat:mat))
  (declare (optimize (speed 3) (space 0) (safety 0))
	   (type boolean enable-optimize?)
	   (type waffetensor out out1))
  (let ((o (decide-out-buffer out x enable-optimize?)))
    (the mgl-mat:mat (mgl-mat:geem! 1.0 x (inv-tensor enable-optimize? out1 y) 0.0 o))))

(defmethod div-tensor (enable-optimize? out out1 x (y mgl-mat:mat))
  (declare (optimize (speed 3) (space 0) (safety 0))
	   (type boolean enable-optimize?)
	   (type waffedatatype x)
	   (type waffetensor out1)
	   (ignore out))
  (unless (= (the fixnum x) 1)
    (error "cl-waffe.backends.mgl-mat: In (!modify a :/= b), a must be tensor or 1."))
  (the mgl-mat:mat (inv-tensor enable-optimize? out1 y)))

(defun dot-tensor (enable-optimize? out x y)
  (declare (ignore enable-optimize? out))
  (mgl-mat:dot (data x) (data y)))

(declaim (ftype (function (boolean waffetensor waffetensor waffetensor) mgl-mat:mat)
		matmul-tensor
		pow-tensor
		compare-tensor
		sum-tensor))
(defun matmul-tensor (enable-optimize? o x y) ; 3D
  (declare (optimize (speed 3) (space 0) (safety 0))
	   (ignore enable-optimize? o)
	   (type boolean enable-optimize?)
	   (type waffetensor o))
  
  (let* ((x1 (ensure-shape :matmul (data x)))
	 (y1 (ensure-shape :matmul (data y)))
	 (transpose-map `(,(typep (data x) 'function) ,(typep (data y) 'function))))
    (unless (or (<= (length (the list (mat-dimensions x1))) 3)
		(<= (length (the list (mat-dimensions y1))) 3))
      (error "cl-waffe.backends.mgl:matmul-tensor Matmul only supports following: 2d * 2d, 2d * 3d, 3d * 2d, 3d * 3d."))

    (let ((x-dims (the list (mat-dimensions x1)))
	  (y-dims (the list (mat-dimensions y1))))

      (cond
	((and (= (length x-dims) 2)
	      (= (length y-dims) 2))
	 (let ((out-dim `(,(if (car transpose-map)
			       (car  (reverse (mat-dimensions x1)))
			       (car  (mat-dimensions x1)))
			  ,(if (second transpose-map)
			       (second (reverse (mat-dimensions y1)))
			       (second (mat-dimensions y1))))))
	   (with-thread-cached-mat (out out-dim :place (gensym)) ; change place name in tensor
	      (matmul-tensor-2d ; do i have to copy?
	       out
	       x1
	       y1
	       (car transpose-map)
	       (second transpose-map)))))
	
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
	   (with-thread-cached-mat (out out-dim
					:displacement 0
					:place (gensym))
	     (dotimes (i (the fixnum (car x-dims)))
	       (reshape-and-displace! out
				      (cdr out-dim)
				      (the fixnum (* i
						     (the fixnum (nth 1 out-dim))
						     (the fixnum (nth 2 out-dim)))))
	      
	       (reshape-and-displace!
		x1
		(cdr shape-first)
		(the fixnum (* i
			       (the fixnum (nth 1 shape-first))
			       (the fixnum (nth 2 shape-first)))))
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
	   (with-thread-cached-mat (out out-dim
					:displacement 0
					:place (gensym))
	     (dotimes (i (the fixnum (car y-dims)))
	       (reshape-and-displace! out
				      (cdr out-dim)
				      (the fixnum
					   (* i
					      (the fixnum
						   (nth 1 out-dim))
					      (the fixnum
						   (nth 2 out-dim)))))
	       (reshape-and-displace! y1
				      (cdr shape-first)
				      (the fixnum
					   (* i
					      (the fixnum
						   (nth 1 shape-first))
					      (the fixnum
						   (nth 2 shape-first)))))
	       (matmul-tensor-2d out x1 y1
				 (car transpose-map)
				 (second transpose-map)))
	     (reshape-and-displace! out out-dim 0)
	     (reshape-and-displace! y1 shape-first displace-first)
	     out)))
	(T (error "cl-waffe.backends.mgl:matmul-tensor: unimplemented combinations."))))))

(declaim (ftype
	  (function
	   (mat mat mat boolean boolean)
	   mat)
	  matmul-tensor-2d))
(defun matmul-tensor-2d (out x y ta? tb?)
  (declare (optimize (speed 3) (space 0) (safety 0)))
  (gemm! 1.0 x y 0 out :transpose-a? ta? :transpose-b? tb?)
  out)

(defun log-tensor (enable-optim out x)
  (declare (optimize (speed 3) (space 0) (safety 0))
	   (type boolean enable-optim)
           (type waffetensor out x))
  (let ((o (decide-out-buffer out x enable-optim)))
           (mgl-mat:.log! o)))

(defun exp-tensor (enable-optim out x)
  (declare (optimize (speed 3) (space 0) (safety 0))
	   (type boolean enable-optim)
           (type waffetensor out x))
  (let ((o (decide-out-buffer out x enable-optim)))
    (mgl-mat:.exp! o)))

(defun sqrt-tensor (enable-optim out x)
  (declare (optimize (speed 3) (space 0) (safety 0))
	   (type boolean enable-optim)
           (type waffetensor out x))
  (let ((o (decide-out-buffer out x enable-optim)))
           (mgl-mat:.sqrt! o)))

(defun pow-tensor (enable-optim out x y)
  (declare (optimize (speed 3) (space 0) (safety 0))
	   (type boolean enable-optim)
           (type waffetensor out x y))
  (let ((o (decide-out-buffer out x enable-optim)))
    (mgl-mat:.expt! o (data y))))

(defun tanh-tensor (enable-optim out x)
  (declare (optimize (speed 3) (space 0) (safety 0))
	   (type boolean enable-optim)
           (type waffetensor out x))
  (let ((o (decide-out-buffer out x enable-optim)))
       (mgl-mat:.tanh! o)))

(defun compare-tensor (enable-optim out x y)
  (declare (optimize (speed 3) (space 0) (safety 0))
	   (type boolean enable-optim)
           (type waffetensor out x y))
  (let ((o (decide-out-buffer out x enable-optim)))
           (mgl-mat:.<! (data y) o)))

(defun sum-tensor (is-first-time-call? out x y)
  (declare (optimize (speed 3) (space 0) (safety 0))
           (type boolean is-first-time-call?)
           (type waffetensor out x y)
	   (ignore is-first-time-call? out))

  (let* ((dims (mgl-mat:mat-dimensions (data x)))
	 (dims (if (and (= 1 (the fixnum (car (last dims))))
			(= 3 (length dims)))
		   (butlast dims)
		   dims))
	 (x1 (mgl-mat:reshape! (mgl-mat:copy-mat (data x)) dims))
	 (dims (case (data y)
		 (1 `(,@(list (car dims)) 1))
		 (0 `(1 ,@(cdr dims)))
		 (T (error "Sum only supports a 2d matrix")))))
    (let ((o (mgl-mat:make-mat dims :initial-element 0)))
      (mgl-mat:sum! x1 o :axis (data y) :beta 1)
      (mgl-mat:reshape! o dims)
      (if (equal dims `(1 1))
	  (mgl-mat:mref o 0 0)
	  o))))

(defun mean-tensor (is-first-time-call? out x y)
  (declare (optimize (speed 3) (space 0) (safety 0))
           (type boolean is-first-time-call?)
           (type waffetensor out x y)
	   (ignore is-first-time-call? out))

  (let* ((dims (mgl-mat:mat-dimensions (data x)))
	 (dims (if (and (= 1 (the fixnum (car (last dims))))
			(= 3 (length dims)))
		   (butlast dims)
		   dims))
	 (x1 (mgl-mat:reshape! (mgl-mat:copy-mat (data x)) dims))
	 (dims (case (data y)
		 (1 `(,@(list (car dims)) 1))
		 (0 `(1 ,@(cdr dims)))
		 (T (error "Sum only supports a 2d matrix")))))
    (let ((o (mgl-mat:make-mat dims :initial-element 0))
	  (s (nth (data y) (mgl-mat:mat-dimensions (data x)))))
      (mgl-mat:sum! x1 o :axis (data y) :beta 1)
      (mgl-mat:scal! (/ 1 (the integer s)) x1)
      (mgl-mat:reshape! o dims)
      (if (equal dims `(1 1))
	  (mgl-mat:mref o 0 0)
	  o))))

(declaim (ftype (function (boolean waffetensor waffetensor waffetensor) mgl-mat:mat) reshape-tensor))
(defun reshape-tensor (enable-optimize out x y)
  (declare (optimize (speed 3) (space 0) (safety 0))
	   (type boolean enable-optimize)
	   (type waffetensor out x y)
	   (ignore enable-optimize out))
  (let ((x1 (mgl-mat:copy-mat (data x))))
    (mgl-mat:reshape! x1 (data y))
    x1))


(define-lisp-kernel (bernoulli-lisp)
    ((mask :mat :io)
     (start-mask fixnum)
     (n fixnum)
     (p single-float))
  (loop for mi upfrom start-mask below (the fixnum (+ start-mask n))
	do (cond ((< (aref mask mi) p)
		  (setf (aref mask mi) 0.0))
		 (t
		  (setf (aref mask mi) 1.0)))))

; having not cuda gpus, i can't test this code lol T_T
(define-cuda-kernel (bernoulli-cuda)
    (void ((mask :mat :io) (n int) (p float)))
  (let ((i (+ (* block-dim-mask block-idx-mask) thread-idx-mask)))
    (when (< i n)
      (if (< (aref x i) p)
	  (set (aref x i) 0.0)
          (set (aref x i) 1.0)))))

(defun bernoulli-tensor (enable-optimize out return-tensor rate)
  (declare (type boolean enable-optimize)
	   (type waffetensor return-tensor x rate))
  (let ((o (decide-out-buffer out return-tensor enable-optimize)))
    (mgl-mat:uniform-random! o)
    (if (use-cuda-p (data return-tensor))
	(progn
	  (print "having not gpus, i've not tested cl-waffe.backends.mgl:bernoulli-tensor yet in cuda.")
	(bernoulli-cuda o (mat-size o) (data rate)
			:grid-dim (list (ceiling (mat-size o) 256) 1 1)
			:block-dim (list 256 1 1)))
	(bernoulli-lisp o (mat-displacement o) (mat-size o) (data rate)))))


;optimize is failed.
(define-lisp-kernel (embedding-forward-lisp)
    ((out :mat :io)
     (x :mat)
     (weights :mat)
     (n fixnum)
     (pad-idx fixnum)
     (embedding-dim fixnum))
  (loop for xi of-type fixnum upfrom 0 below n
	do (cond
	     ((= pad-idx (round (aref x xi)))
	      nil)
	     (T
	      (loop for ei of-type fixnum upfrom 0 below embedding-dim
		    do (setf (aref out (+ (the fixnum
					       (* (the fixnum embedding-dim) xi))
					  ei))
			     (aref weights
				     (+ ei
					(the fixnum (* (round (aref x xi))
						       (the fixnum embedding-dim)))))))))))

(defun embedding-forward (enable-optimize x weights pad-idx)
  "(with-searching-calc-node :embedding-forward x weights pad-idx) -> embeddings"
  (declare (type boolean enable-optimize)
	   (type waffetensor out x weights pad-idx)
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

(declaim (ftype (function (keyword boolean waffetensor waffetensor cons) (or mgl-mat:mat cl-waffe:waffedatatype)) dispatch-kernel))
(defun dispatch-kernel (function is-first-time-call? destructable-tensor destructable-tensor1 args)
  (declare (optimize (speed 3) (space 0) (safety 0))
	   (type keyword function)
	   (type boolean is-first-time-call?)
	   (type waffetensor destructable-tensor)
	   (type cons args))
  (case function
    (:add     (add-tensor is-first-time-call? destructable-tensor destructable-tensor1 (data (car args)) (data (second args))))
    (:sub     (sub-tensor is-first-time-call? destructable-tensor destructable-tensor1 (data (car args)) (data (second args))))
    (:mul     (mul-tensor is-first-time-call? destructable-tensor destructable-tensor1 (data (car args)) (data (second args))))
    (:div     (div-tensor is-first-time-call? destructable-tensor destructable-tensor1 (data (car args)) (data (second args))))
    (:dot     (dot-tensor is-first-time-call? destructable-tensor (car args) (second args)))
    (:matmul  (matmul-tensor is-first-time-call? destructable-tensor (car args) (second args)))
    (:log     (log-tensor is-first-time-call? destructable-tensor (car args)))
    (:exp     (exp-tensor is-first-time-call? destructable-tensor (car args)))
    (:pow     (pow-tensor is-first-time-call? destructable-tensor (car args) (second args)))
    (:sqrt    (sqrt-tensor is-first-time-call? destructable-tensor (car args)))
    (:sum     (sum-tensor is-first-time-call? destructable-tensor (car args) (second args)))
    (:mean    (mean-tensor is-first-time-call? destructable-tensor (car args) (second args)))
    (:tanh    (tanh-tensor is-first-time-call? destructable-tensor (car args)))
    (:reshape (reshape-tensor is-first-time-call? destructable-tensor (car args) (second args)))
    (:<       (compare-tensor is-first-time-call? destructable-tensor (car args) (second args)))
    (:repeat  (mgl-repeat (data (car args)) (data (third args)) :axis (data (second args))))
    (:bernoulli (bernoulli-tensor is-first-time-call? destructable-tensor (car args) (second args)))
    (:transpose (deliv-delay (data (car args)) #'mgl-mat:transpose))
    (:embedding-forward (embedding-forward nil (car args)
					   (second args)
					   (third args)))
    (:embedding-backward (embedding-backward nil
					     (car args)
					     (second args)
					     (third args)
					     (fourth args)))
    (T (error "~a is not yet implemented" function))))

