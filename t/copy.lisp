
(in-package :cl-waffe-test)

(in-suite :test)

#|
Testing !aref, (setf !aref) for multi dims
|#

(defun use-faref (tensor &rest dims)
  (apply #'cl-waffe::%faref tensor dims))

(defun use-saref (tensor &rest dims)
  (apply #'cl-waffe::%saref nil tensor dims))

(defun use-setf-faref (target tensor &rest dims)
  (apply #'cl-waffe::%write-faref target tensor dims))

(defun use-setf-saref (target tensor &rest dims)
  (apply #'cl-waffe::%saref target tensor dims))

(defmacro arefs-test (tensor &rest dims &aux (r1 (gensym)) (r2 (gensym)))
  `(let ((,r1 (use-faref ,tensor ,@dims))
	 (,r2 (use-saref ,tensor ,@dims)))
     (M= (data ,r1)
	 (data ,r2))))

(defmacro setf-aref-test1 (tensor
			   &rest dims
			   &aux
			     (storeroom (gensym))
			     (x (gensym))
			     (y (gensym)))
  `(let ((,storeroom (copy-mat (data ,tensor)))
	 (,x (copy-mat (data ,tensor)))
	 (,y (copy-mat (data ,tensor))))
     (use-setf-faref (const ,x)
		     (!aref (const ,x) ,@dims)
		     ,@dims)

     (use-setf-saref (const ,x)
		     (!aref (const ,x) ,@dims)
		     ,@dims)
     (and (M= ,x ,y)
	  (M= ,storeroom ,x))))

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

; broadcasting errorは、!arefされてないoutと!arefされたTensorを比べてるから起きる

(test setf-aref-test-1d
      (is (setf-aref-test1 (!randn `(10)) t))
      (is (setf-aref-test1 (!randn `(10)) 0))
      (is (setf-aref-test1 (!randn `(10)) 1))
      (is (setf-aref-test1 (!randn `(10)) -1))
      (is (setf-aref-test1 (!randn `(10)) -2))
      (is (setf-aref-test1 (!randn `(10)) '(0 1)))
      (is (setf-aref-test1 (!randn `(10)) '(1 2)))
      (is (setf-aref-test1 (!randn `(10)) '(2 -1))))

(test setf-aref-test-2d
      (is (setf-aref-test1 (!randn `(10 10)) t))
      (is (setf-aref-test1 (!randn `(10 10)) 0 0))
      (is (setf-aref-test1 (!randn `(10 10)) t 0))
      (is (setf-aref-test1 (!randn `(10 10)) 0 t))
      (is (setf-aref-test1 (!randn `(10 10)) 1 t))
      (is (setf-aref-test1 (!randn `(10 10)) t 1))

      (is (setf-aref-test1 (!randn `(10 10)) '(0 3) t))
      (is (setf-aref-test1 (!randn `(10 10)) '(3 -1) t))
      (is (setf-aref-test1 (!randn `(10 10)) '(1 3) t))
      (is (setf-aref-test1 (!randn `(10 10)) '(1 3) '(1 3))))

(test setf-aref-test-3d
      (is (setf-aref-test1 (!randn `(10 10 10)))


