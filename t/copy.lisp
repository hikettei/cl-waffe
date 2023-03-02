
(in-package :cl-waffe-test)

(in-suite :test)

(defun use-faref (tensor &rest dims)
  (apply #'cl-waffe::%faref tensor dims))

(defun use-saref (tensor &rest dims)
  (apply #'cl-waffe::%saref nil tensor dims))

(defmacro arefs-test (tensor &rest dims &aux (r1 (gensym)) (r2 (gensym)))
  `(let ((,r1 (use-faref ,tensor ,@dims))
	 (,r2 (use-saref ,tensor ,@dims)))
     (M= (data ,r1)
	 (data ,r2))))

(defmacro setf-aref-test (input
			  tensor
			  &rest dims
			  &aux (storeroom (gensym)))
  `(let ((,storeroom (copy-mat (data tensor))))
     
     ))

(defparameter aref-arg1 (!ones `(10 10)))
(defparameter aref-arg2 (!ones `(100 10 10)))
(defparameter aref-arg3 (!ones `(100 100 10 10)))


(test aref-test-2d
      (is (arefs-test aref-arg1 t t))
      
      (is (arefs-test aref-arg1 0 t))
      (is (arefs-test aref-arg1 1 t))

      (is (arefs-test aref-arg1 -1 t))
      (is (arefs-test aref-arg1 -2 t))

      (is (arefs-test aref-arg1 0 1))
      (is (arefs-test aref-arg1 0 2))

      (is (arefs-test aref-arg1 3 4))
      (is (arefs-test aref-arg1 -1 -2))
      
      (is (arefs-test aref-arg1 '(0 3) t))
      (is (arefs-test aref-arg1 '(0 3) '(0 3)))
      (is (arefs-test aref-arg1 '(2 3) '(3 4)))

      (is (arefs-test aref-arg1 '(2 -1) '(3 -1)))

      ;(is (arefs-test aref-arg1 '(0 t) '(0 -1))) APIs are different...
      ;(is (arefs-test aref-arg1 '(1 t) '(-1 t)))
      )

(test aref-test-3d
      (is (arefs-test aref-arg2 t t t))
      (is (arefs-test aref-arg2 0 0 0))
      (is (arefs-test aref-arg2 1 1 1))
      (is (arefs-test aref-arg2 0 1 0))
      (is (arefs-test aref-arg2 1 2 3))
      (is (arefs-test aref-arg2 1 t 3))
      (is (arefs-test aref-arg2 t 1 1))
      (is (arefs-test aref-arg2 1 t t))

      (is (arefs-test aref-arg2 0 '(0 3) t))
      (is (arefs-test aref-arg2 0 '(4 5) t))
      (is (arefs-test aref-arg2 '(0 3) '(1 3) '(2 3))))

(test aref-test-4d
      (is (arefs-test aref-arg3 t t t t))
      (is (arefs-test aref-arg3 0 0 0 0))
      (is (arefs-test aref-arg3 1 t 1 t))
      (is (arefs-test aref-arg3 t t t 1))
     
      (is (arefs-test aref-arg3 1 2 3 4))

      (is (arefs-test aref-arg3 '(0 3) '(0 3) '(0 3) '(0 3)))
      (is (arefs-test aref-arg3 '(0 -1) '(1 -1) '(2 -2) '(3 -1))))
