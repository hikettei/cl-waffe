
(in-package :cl-waffe.backends.mgl)

; Todo Rewrite with define-lisp-kernel

(defparameter *copyed-num* 0)

(defmacro duplicate-tensor (mat)
  `(let ((o (mgl-mat:make-mat (mgl-mat:mat-dimensions ,mat))))
     (incf *copyed-num* 1)
     (print *copyed-num*)
     (mgl-mat:copy! ,mat o)
     o))

(defmacro apply-destruct (out tensor)
  `(progn
     (print "destructed!")
     (setf (waffetensor-report-index ,tensor)
	   (waffetensor-report-index ,out)) ;dop(a,b)a=a, wait for refresh
     (setf (waffetensor-is-data-destructed? ,tensor) t)
     (incf (waffetensor-destructively-calln ,tensor) 1)))

(defmacro assure-destructed? (out tensor)
  `(progn
     (unless (waffetensor-destructive? ,tensor)
       (error "Kernel Error: Modifying tensor that is not allowed to destruct"))
     ,out))

(defmacro decide-out-buffer (out args var enable-optim)
  `(progn
     (if ,out
	 (progn
	   (apply-destruct ,out ,var)
	   (if ,enable-optim
	       (assure-destructed? (data ,out) ,var)
	       (duplicate-tensor ,args)))
	 (duplicate-tensor ,args))))

(defun repeat (array n &key axis)
  (if (typep array 'mgl-mat:mat)
      (if axis
	  (if (>= (length (mgl-mat:mat-dimensions array)) 2)
              (mgl-mat:stack axis (loop for i below n collect array))
	      (repeat (mgl-mat:reshape array `(,@(mgl-mat:mat-dimensions array) 1)) n :axis axis))
	  (error "axis=-1"))
      (error "array != mat")))

(defun mgl-full-like (tensor value)
  (mgl-mat:make-mat (mgl-mat:mat-dimensions tensor)
		    :initial-element value))

(defun transposed-mgl-full-like (tensor value)
  (let ((dims (mgl-mat:mat-dimensions tensor)))
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

(defun ensure-shape (ope args)
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

(defun kernel (ope args out variables enable-optim)
  ;(declare (optimize (speed 3) (space 0) (safety 0) (debug 0)))
  (if (and (find ope `(:mul :div :matmul))
	   (find t (map 'list (lambda (x) (if (and (not (typep x 'mgl-mat:mat)) (not (typep x 'function))) (= x 1))) args)))
      (if (and (eq ope :div) (= (car args) 1))
	  (let ((o (decide-out-buffer out (second args) (second variables) enable-optim)))
	    (mgl-mat:.inv! o))
	  (find 1 args :test (lambda (x y) (declare (ignore x)) (typep y 'mgl-mat:mat))))
      (let ((transpose-map (map 'list (lambda (x) (typep x 'function)) args))
	    (args (ensure-shape ope args)))
    (case ope
      (:add (if (eq (mgl-mat:mat-dimensions (car args)) (mgl-mat:mat-dimensions (second args)))
		(let ((o (decide-out-buffer out (second args) (second args) enable-optim)))
		  (mgl-mat:axpy! 1.0 (car args) o))
		(mgl-mat:m+ (car args) (second args)))) ;slow
      (:sub (if (eq (mgl-mat:mat-dimensions (car args)) (mgl-mat:mat-dimensions (second args)))
		(let ((o (decide-out-buffer out (car args) (second args) enable-optim)))
		  (mgl-mat:axpy! -1.0 (second args) o))
		(mgl-mat:m- (car args) (second args)))) ;slow
      (:mul (let ((out (mgl-mat:make-mat (mgl-mat:mat-dimensions (car args)))))
	      (mgl-mat:geem! 1 (car args) (second args) 0 out)
	      out))
      (:div (let* ((out (mgl-mat:make-mat (mgl-mat:mat-dimensions (car args))))
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
		(error "cl-waffe.backends.mgl: :dot dotproduct failed due to unsatisfied with (!dims a) <= 2 and (!dims b) <= 2"))
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
