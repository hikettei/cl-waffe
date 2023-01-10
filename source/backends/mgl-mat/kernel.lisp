
(in-package :cl-waffe.backends.mgl)

; Todo Rewrite with define-lisp-kernel

(defmacro duplicate-tensor (mat)
  `(let ((o (mgl-mat:make-mat (mgl-mat:mat-dimensions ,mat))))
     (mgl-mat:copy! ,mat o)
     o))

(defmacro apply-destruct (out)
  `(progn ; tensor=out
     (setf (waffetensor-is-data-destructed? ,out) t)
     (setf (waffetensor-destructively-calln ,out) 1)))

(defmacro assure-destructed? (out tensor)
  `(progn
     (unless (waffetensor-destructive? ,tensor)
       (error "Kernel Error: Modifying tensor that is not allowed to destruct"))
     ,out))

(defmacro decide-out-buffer (out args var enable-optim)
  `(progn
     ;(apply-destruct ,var)
     (if ,out
	 (progn
	   (if (and ,enable-optim (waffetensor-destructive? ,out))
	       (assure-destructed? (data ,out) ,var)
	       (duplicate-tensor (assure-destructed? ,args ,var))))
	 (duplicate-tensor (assure-destructed? ,args ,var)))))

(declaim (ftype (function (mgl-mat:mat fixnum) mgl-mat:mat) repeat))
(defun repeat (tensor n &key axis)
  (declaim (optimize (speed 3) (safety 0) (debug 0))
	   (type mgl-mat:mat tensor)
	   (type fixnum n axis))
  (if (typep tensor 'mgl-mat:mat)
      (if axis
	  (if (>= (length (mgl-mat:mat-dimensions tensor)) 2)
              (mgl-mat:stack axis (loop for i below n collect tensor))
	      (repeat (mgl-mat:reshape tensor `(,@(mgl-mat:mat-dimensions tensor) 1)) n :axis axis))
	  (error "axis=-1"))
      (error "array != mat")))

(declaim (ftype (function (mgl-mat:mat waffesupporteddatatype) mgl-mat:mat) trasposedmgl-full-like mgl-full-like))

(defun mgl-full-like (tensor value)
  (declare (optimize (speed 3) (safety 0) (debug 0))
	   (type mgl-mat:mat tensor)
	   (type waffesupporteddatatype value))
  (mgl-mat:make-mat (mgl-mat:mat-dimensions tensor)
		    :initial-element value))

(defun transposed-mgl-full-like (tensor value)
  (declare (optimize (speed 3) (safety 0) (debug 0))
	   (type mgl-mat:mat tensor)
	   (type waffesupporteddatatype value))
  (let ((dims (mgl-mat:mat-dimensions tensor)))
    (declare (type cons dims))
    (mgl-mat:make-mat (reverse dims)
		      :initial-element value)))

(defparameter *v2v-operations* `(:add :sub :mul :div :dot :matmul))
(defparameter *abort-delay-instruction* :matmul)

(defmacro deliv-delay (tensor func &rest args)
  `(lambda (shape? step?)
     (if shape?
	 (reverse (mgl-mat:mat-dimensions ,tensor))
	 (if step?
	     (funcall ,func ,tensor ,@args) ; receive before node
	     ,tensor)))) ; abort before node

(defmacro next-delay (delay state)
  `(if (typep ,delay 'function)
       (funcall ,delay nil ,state)
       ,delay))

(defmacro abort-delay (delay)
  `(next-delay ,delay nil))

(defmacro receive-delay (delay)
  `(next-delay ,delay t))

(declaim (ftype (function (keyword cons) cons) ensure-shape))
(defun ensure-shape (ope args)
  (declare (optimize (speed 3) (safety 0) (debug 0))
	   (type keyword ope)
	   (type cons args))
  (if (find ope *v2v-operations*)
      (let ((m (find 'mgl-mat:mat args :test (lambda (x y) (typep y x)))))
	(unless m (error "Waffe Kernel Error"))
	(map 'list (lambda (x)
		     (if (or (typep x 'mgl-mat:mat) (typep x 'function))
			 (if (eq ope *abort-delay-instruction*)
			     (abort-delay x)
			     (receive-delay x))
			 (if (eq ope :matmul)
			     (transposed-mgl-full-like m x)
			     (mgl-full-like m x))))
	     args))
      ; suppose that args has at least 1 mats.
      args))

(declaim (ftype (function (keyword cons null cons boolean) (or mgl-mat:mat waffesupporteddatatype)) kernel))
(defun kernel (ope args out variables enable-optim)
  (declare (optimize (speed 3) (safety 0) (debug 0))
	   (type keyword ope)
	   (type cons args variables)
	   (type null out)
	   (type boolean enable-optim))
  (if (and (find ope `(:mul :div :matmul))
	   (find t (map 'list (lambda (x) (if (and (not (typep x 'mgl-mat:mat)) (not (typep x 'function))) (= x 1))) args)))
      (if (and (eq ope :div) (= (car args) 1)) ; 1/tensor
	  (let ((o (decide-out-buffer out (second args) (second variables) enable-optim)))
	    (mgl-mat:.inv! o))
	  (find 1 args :test (lambda (x y) (declare (ignore x)) (typep y 'mgl-mat:mat))))
      (let ((transpose-map (map 'list (lambda (x) (typep x 'function)) args))
	    (args (ensure-shape ope args)))
    (case ope
      (:add (if (eq (mgl-mat:mat-dimensions (car args)) (mgl-mat:mat-dimensions (second args)))
		(let ((o (mgl-mat:make-mat (mgl-mat:mat-dimensions (car args)))))
		  (mgl-mat:axpy! 1.0 (car args) o)
		  (mgl-mat:axpy! 1.0 (second args) o)
		  o)
		(mgl-mat:m+ (car args) (second args)))) ;slow
      (:sub (if (eq (mgl-mat:mat-dimensions (car args)) (mgl-mat:mat-dimensions (second args)))
		(let ((o (mgl-mat:make-mat (mgl-mat:mat-dimensions (car args)))))
		  (mgl-mat:axpy! -1.0 (second args) o)
		  (mgl-mat:axpy! 1.0 (car args) o)
		  o)
		(mgl-mat:m- (car args) (second args)))) ;slow
      (:mul (let ((out (mgl-mat:make-mat (mgl-mat:mat-dimensions (car args)))))
	      (mgl-mat:geem! 1 (car args) (second args) 0 out)
	      out))
      (:div (let* ((out       (mgl-mat:make-mat (mgl-mat:mat-dimensions (car args))))
		   (args-copy (mgl-mat:copy-mat (second args)))
		  ;initilizing new mats...
		   (inv (mgl-mat:.inv! args-copy)))
	      (mgl-mat:geem! 1 (car args) inv 0 out)
	      out))
      (:dot (mgl-mat:dot (car args) (second args))) ;slow
      (:matmul (let ((out (mgl-mat:make-mat `(,(if (car transpose-map)
						   (car  (reverse (mgl-mat:mat-dimensions (car args))))
						   (car  (mgl-mat:mat-dimensions (car args))))
					      ,(if (second transpose-map)
						   (second (reverse (mgl-mat:mat-dimensions (second args))))
						   (second (mgl-mat:mat-dimensions (second args))))))))
	      (unless (and (<= (length (mgl-mat:mat-dimensions (car args))) 2)
			   (<= (length (mgl-mat:mat-dimensions (second args))) 2))
		(error "cl-waffe.backends.mgl: :matmul failed due to unsatisfied with (!dims a) <= 2 and (!dims b) <= 2"))
		 (mgl-mat:gemm! 1 (car args) (second args) 0 out :transpose-a? (car transpose-map) :transpose-b? (second transpose-map))
		 out))
      (:log (let ((o (decide-out-buffer out (car args) (car variables) enable-optim)))
	      (mgl-mat:.log! o)))
      (:exp (let ((o (decide-out-buffer out (car args) (car variables) enable-optim)))
	      (mgl-mat:.exp! o)))
      (:pow (let ((o (decide-out-buffer out (car args) (car variables) enable-optim)))
	      (mgl-mat:.expt! o (second args))))
      (:sqrt (let ((o (decide-out-buffer out (car args) (car variables) enable-optim)))
	       (mgl-mat:.sqrt! o)))
      (:sum  (let* ((dims (mgl-mat:mat-dimensions (car args)))
		    (dims (if (and (= 1 (car (last dims)))
				   (= 3 (length dims)))
			      (butlast dims)
			      dims))
		    (x (mgl-mat:reshape! (mgl-mat:copy-mat (car args)) dims))
		    (dims (case (second args)
			   (1 `(,@(list (car dims)) 1))
			   (0 `(1 ,@(cdr dims)))
			   (T (error "Sum only supports a 2d matrix")))))
	       (let ((o (mgl-mat:make-mat dims :initial-element 0)))
		 (mgl-mat:sum! x o :axis (second args) :beta 1)
		 (mgl-mat:reshape! o dims)
		 (if (equal dims `(1 1))
		     (mgl-mat:mref o 0 0)
		     o))))
      (:mean (let ((result (numcl:mean (mat-to-numcl (car args)) :axes (second args))))
               ;Using CPU but fast enough as long as used for losses
	       (if (numcl:numcl-array-p result)
		   (numcl-to-mat result)
		   result)))
      (:tanh (let ((o (decide-out-buffer out (car args) (car variables) enable-optim)))
	       (mgl-mat:.tanh! o)))
      (:reshape (let ((x (decide-out-buffer out (car args) (car variables) enable-optim)))
		    (mgl-mat:reshape! x (second args))
		  x))
      (:< (let ((x (decide-out-buffer out (car args) (car variables) enable-optim)))
	       (mgl-mat:.<! (second args) x)
	    x))
    ;  (:> (let ((x (decide-out-buffer out (car args))))
	 ;   (mgl-mat:.<! x (second args)) ; x = (second args) ...?
	  ;  x))
      (:repeat (repeat (car args) (third args) :axis (second args)))
      (:transpose (deliv-delay (car args) mgl-mat:transpose)) ; second args? please remain that this makes superrr slow
      (T (error "~a is nt yet implemented" ope))))))

(defun infomation ())
