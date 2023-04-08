
(in-package :cl-waffe.kernel)

(defun dtype-blas-prefix ()
  (dtypecase
   (:short "?")
   (:float "s")
   (:double "d")
   (T (error "cl-waffe.kernel doesn't support the dtype."))))

(defun blas-function-name (name)
  (format nil "~A~A_"
	  (dtype-blas-prefix)
	  name))

(defmacro define-blas-function (lisp-name cname return-type &rest args)
  `(define-with-typevar ,lisp-name utype (,@(map 'list #'second args))
     (foreign-funcall (blas-function-name ,cname)
		      ,@(map 'list #'(lambda (arg)
				       (case arg
					 (:mat
					  (list :pointer *dtype*))
					 (:float *dtype*)
					 (T arg)))
			     (flatten args))
		      ,(case return-type
			 (:mat (list :pointer *dtype*))
			 (:float *dtype*)
			 (T return-type)))))

(define-blas-function blas-axpy "axpy" :mat
		      (:int n)
		      (:float alpha)
		      (:mat x)
		      (:int incx)
		      (:mat y)
		      (:int incy))
