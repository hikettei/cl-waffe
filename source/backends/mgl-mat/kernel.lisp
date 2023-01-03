
(in-package :cl-waffe.backends.mgl)

; Todo Rewrite with define-lisp-kernel

(defmacro avoid-destructive (ope mat &rest args)
  `(let ((result (mgl-mat:copy-mat ,mat)))
     ,(unless (null args)
	 `(,ope result ,@args)
	 `(,ope result))
     result))

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

(defparameter *v2v-operations* `(:add :sub :mul :div))

(defun ensure-shape (ope args)
  (if (find ope *v2v-operations*)
      (let ((m (find 'mgl-mat:mat args :test (lambda (x y) (typep y x)))))
	(unless m (error ""))
	(map 'list (lambda (x)
		     (if (typep x 'mgl-mat:mat)
			 x
			 (mgl-full-like m x)))
	     args))
      ; suppose that args has at least 1 mats.
      args))

(defun numcl-to-mat (ncl)
  (mgl-mat:array-to-mat ncl))

(defun kernel (ope args) ; operations with CPU is ridiculously slow... So I need to rewrite it with define-lisp-kernel/define-cuda-kernel

  (let* ((args (ensure-shape ope args)))
    (case ope
      (:add (mgl-mat:M+ (car args) (second args)))
      (:sub (mgl-mat:M- (car args) (second args)))
      (:mul (let ((out (mgl-mat:make-mat (mgl-mat:mat-dimensions (car args))
					 :initial-element 0)))
	      (mgl-mat:geem! 1 (car args) (second args) 0 out)))
      (:div (let ((out (mgl-mat:make-mat (mgl-mat:mat-dimensions (car args))
					 :initial-element 0))
		  (inv (avoid-destructive mgl-mat:.inv! (second args))))
	      (mgl-mat:geem! 1 (car args) inv 0 out)))
      (:dot (let ((out (mgl-mat:make-mat `(,(car (mgl-mat:mat-dimensions (car args)))
					   ,(second (mgl-mat:mat-dimensions (second args))))
					 :initial-element 0)))
	      (mgl-mat:gemm! 1 (car args) (second args) 0 out)))
      (:log (avoid-destructive mgl-mat:.log! (car args)))
      (:exp (avoid-destructive mgl-mat:.exp! (car args)))
      (:pow (avoid-destructive mgl-mat:.expt! (car args) (second args)))
      (:sum  (numcl-to-mat (numcl:sum  (mat-to-numcl (car args)) :axes (second args)))) ; CPU
      (:mean (numcl-to-mat (numcl:mean (mat-to-numcl (car args)) :axes (second args)))) ;CPU
      (:tanh (avoid-destructive mgl-mat:.tanh! (car args)))
      (:reshape   (numcl-to-mat (numcl:reshape (mat-to-numcl (car args)) (second args)))) ;CPU
      (:repeat    (numcl-to-mat (repeat (mat-to-numcl (car args)) (third args) :axis (second args)))) ;CPU and esp slow
      (:transpose (mgl-mat:transpose (car args))) ; second args?
      (T (error "~a is nt yet implemented" ope)))))

(defun infomation ())
