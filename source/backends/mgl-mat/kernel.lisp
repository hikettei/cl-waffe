
(in-package :cl-waffe.backends.mgl)

; Todo Rewrite with define-lisp-kernel

(defmacro decide-out-buffer (out args)
  `(progn
     (if ,out
	 (progn
	   (mgl-mat:copy! ,args ,out)
	   ,out)
	 (progn
	   (mgl-mat:copy-mat ,args)))))
      

(defun repeat (array n &key axis)
  (if (numcl:arrayp array)
      (if axis
          (numcl:concatenate (make-list n :initial-element array) :axis axis)
          (numcl:flatten
           (numcl:concatenate (make-list n :initial-element (numcl:reshape array `(,@(numcl:shape array) -1))) :axis -1)))
      (progn
        ;(assert (null axis))
        (numcl:full n array))))

(defun mgl-full-like (tensor value)
  (mgl-mat:make-mat (mgl-mat:mat-dimensions tensor)
		    :initial-element value))

(defun mat-to-numcl (mat)
  (numcl:asarray (mgl-mat:mat-to-array mat)))

(defparameter *v2v-operations* `(:add :sub :mul :div :dot :matmul))

(defun ensure-shape (ope args)
  (if (find ope *v2v-operations*)
      (let ((m (find 'mgl-mat:mat args :test (lambda (x y) (typep y x)))))
	(unless m (error "Waffe Kernel Error"))
	(map 'list (lambda (x)
		     (if (typep x 'mgl-mat:mat)
			 x
			 (if (eq ope :matmul)
			     (mgl-mat:transpose (mgl-full-like m x)) ; bottle neck...
			     (mgl-full-like m x))))
	     args))
      ; suppose that args has at least 1 mats.
      args))

(defun numcl-to-mat (ncl)
  (mgl-mat:array-to-mat ncl))

(defun kernel (ope args &optional out) ; operations with CPU is ridiculously slow... So I need to rewrite it with define-lisp-kernel/define-cuda-kernel
  (if (and (find ope `(:mul :div :matmul))
	   (find t (map 'list (lambda (x) (if (not (typep x 'mgl-mat:mat)) (= x 1))) args)))
      (if (and (eq ope :div) (= (car args) 1))
	  (let ((o (decide-out-buffer out (second args))))
	    (mgl-mat:.inv! o))
	  (find 1 args :test (lambda (x y) (declare (ignore x)) (typep y 'mgl-mat:mat))))
  (let* ((args (ensure-shape ope args)))
    (case ope
      (:add (mgl-mat:M+ (car args) (second args)))
      (:sub (mgl-mat:M- (car args) (second args)))
      (:mul (let ((out (if out
			   out
			   (mgl-mat:make-mat (mgl-mat:mat-dimensions (car args))
					     :initial-element 0))))
	      (mgl-mat:geem! 1 (car args) (second args) 0 out)
	      out))
      (:div (let* ((out (if out
			   out
			   (mgl-mat:make-mat (mgl-mat:mat-dimensions (car args))
					     :initial-element 0)))
		  (args-copy (mgl-mat:copy-mat (second args)))
		  ;initilizing new mats...
		  (inv (mgl-mat:.inv! args-copy)))
	      (mgl-mat:geem! 1 (car args) inv 0 out)
	      out))
      (:dot (mgl-mat:dot (car args) (second args)))
      (:matmul (let ((out (if out
			      out
			      (mgl-mat:make-mat `(,(car (mgl-mat:mat-dimensions (car args)))
					        ,(second (mgl-mat:mat-dimensions (second args))))
					        :initial-element 0))))
	      (unless (and (<= (length (mgl-mat:mat-dimensions (car args))) 2)
			   (<= (length (mgl-mat:mat-dimensions (second args))) 2))

		(error "cl-waffe.backends.mgl: :dot DotProduct Failed due to unsatisfication with (!dims A) <= 2 and (!dims B) <= 2"))
		 (mgl-mat:gemm! 1 (car args) (second args) 0 out)
		 out))
      (:log (let ((o (decide-out-buffer out (car args))))
	      (mgl-mat:.log! o)))
      (:exp (let ((o (decide-out-buffer out (car args))))
		(mgl-mat:.exp! o)))
      (:pow (if out
		(progn
		  (mgl-mat:copy! (car args) out)
		  (mgl-mat:.expt! out (second args))
		  out)
		(let ((x (mgl-mat:copy-mat (car args))))
		  (mgl-mat:.expt! x (second args))
		  x)))
      (:sum  (let ((result (numcl:sum  (mat-to-numcl (car args)) :axes (second args))))
	       ;CPU
	       (if (typep result 'fixnum)
		   result
		   (numcl-to-mat result))))
      (:mean (let ((result (numcl:mean (mat-to-numcl (car args)) :axes (second args))))
               ;CPU
	       (if (typep result 'fixnum)
		   result
		   (numcl-to-mat result))))
      (:tanh (let ((o (decide-out-buffer out (car args))))
	       (mgl-mat:.tanh! o)))
      (:reshape (let ((x (mgl-mat:copy-mat (car args)))) ;displaceベースに書き換える
		    (mgl-mat:reshape! x (second args))
		    x))
      (:repeat (let ((result (numcl-to-mat (repeat (mat-to-numcl (car args)) (third args) :axis (second args)))))
                    ;esp its too slow, rewrite with stack
		    result))
      (:transpose (mgl-mat:transpose (car args))) ; second args?
      (T (error "~a is nt yet implemented" ope))))))

(defun infomation ())
