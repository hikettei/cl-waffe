
(in-package :cl-waffe-test)

(in-suite :test)

#|
  Testing broadcasted operators.
|#

(defparameter arg1 (!randn `(1000 1000)))
(defparameter arg2 (!randn `(1000 1)))

(defun broadcast-1 (func a b)
  (multiple-value-bind (a b) (cl-waffe::straighten-up a b)
			(cl-waffe.backends.mgl::broadcasting-apply-facet func a b)))

(defun broadcast-2 (func a b)
  (multiple-value-bind (a b) (cl-waffe::straighten-up a b)
    (cl-waffe.backends.mgl::broadcasting-apply-mgl func a b)))

(defun broadcast-3 (func a b)
  (multiple-value-bind (a b) (cl-waffe::straighten-up a b)
    (cl-waffe.backends.mgl::%broadcasting-single-float-cpu func a b)))

(defun brc (ope a b)
  (case ope
    (:add
     (let ((x1 (broadcast-1 :+ a b))
	   (x2 (broadcast-1 :+ a b))
	   (x3 (broadcast-1 :+ a b)))
       (if (and (M= x1 x2)
		(M= x1 x3))
	   (const x1)
	   (error "The result of broadcasting didn't match."))))
    (:sub
     (let ((x1 (broadcast-1 :- a b))
	   (x2 (broadcast-1 :- a b))
	   (x3 (broadcast-1 :- a b)))
       (if (and (M= x1 x2)
		(M= x1 x3))
	   (const x1)
	   (error "The result of broadcasting didn't match."))))
    (:mul
     (let ((x1 (broadcast-1 :* a b))
	   (x2 (broadcast-1 :* a b))
	   (x3 (broadcast-1 :* a b)))
       (if (and (M= x1 x2)
		(M= x1 x3))
	   (const x1)
	   (error "The result of broadcasting didn't match."))))))

(defun broadcast-test (a b)
  (and
   (brc :add a b)
   (brc :add b a)
   (brc :sub a b)
   (brc :sub b a)
   (brc :mul a b)
   (brc :mul b a)))

; most basic
(defun simple-test1 ()
  (format t "Test1: broadcasting (!add arg1 arg2)...")
  (let* ((arg2-r (!repeats arg2 1 1000))
	 (result (time (brc :add arg1 arg2)))
	 (result1 (time (!add arg1 arg2-r))))
    (M= (value result) (value result1))))

; Swap args
(defun simple-test2 ()
  (format t "Test2: broadcasting (!add arg2 arg1)...")
  (let* ((arg2-r (!repeats arg2 1 1000))
	 (result (time (brc :add arg2 arg1)))
	 (result1 (time (!add arg1 arg2-r))))
    (M= (value result) (value result1))))

; For other operators
(defun simple-test3 ()
  (format t "Test3: broadcasting (!sub arg2 arg1)...")
  (let* ((arg2-r (!repeats arg2 1 1000))
	 (result (time (brc :sub arg1 arg2)))
	 (result1 (time (!sub arg1 arg2-r))))
    (M= (value result) (value result1))))

; For other operators
(defun simple-test4 ()
  (format t "Test4: broadcasting (!mul arg2 arg1)...")
  (let* ((arg2-r (!repeats arg2 1 1000))
	 (result (time (brc :mul arg2 arg1)))
	 (result1 (time (!mul arg1 arg2-r))))
    (M= (value result) (value result1))))

; Testing unsqueeze
(defun simple-test5 ()
  (format t "Test5:")
  (let* ((k 1.0)
	 (k-tensor (!fill `(1) k))
	 (result  (time (brc :add arg1 k-tensor)))
	 (result1 (time (!add arg1 k))))
    (M= (value result) (value result1))))

; For 3D
(defun simple-test6 ()
  (format t "Test6:")
  (let* ((m (!randn `(100 100 1)))
	 (k (!repeats m 2 100))
	 (l (!randn `(100 100 100)))
	 (result  (time (brc :add m l)))
	 (result1 (time (brc :add k l))))
    (brc :sub m l)
    (brc :mul m l)
    (brc :sub k l)
    (brc :mul k l)
    (M= (value result) (value result1))))


(test broadcasting-test
      (is (simple-test1))
      (is (simple-test2))
      (is (simple-test3))
      (is (simple-test4))
      (is (simple-test5))
      (is (simple-test6)))

; Tests below supposed broadcasting-apply-facet to be OK for all operations.

(test broadcasting-blas-squares
      (is (broadcast-test (!randn `(10 10)) (!randn `(10 10))))
      (is (broadcast-test (!randn `(10 10 10)) (!randn `(10 10 10))))
      (is (broadcast-test (!randn `(10 10 10 10)) (!randn `(10 10 10 10)))))

; All combines of broadcasting

(test broadcasting-blas-2d-repeats
      (is (broadcast-test (!randn `(10 1)) (!randn `(10 10))))
      (is (broadcast-test (!randn `(10 10)) (!randn `(10 1))))
      
      (is (broadcast-test (!randn `(1 10)) (!randn `(10 10))))
      (is (broadcast-test (!randn `(10 10)) (!randn `(1 10))))

      (is (broadcast-test (!randn `(10 1)) (!randn `(1 10))))
      (is (broadcast-test (!randn `(1 10)) (!randn `(10 1)))))

(test broadcasting-blas-1d
      (is (broadcast-test (!randn `(10)) (!randn `(1))))
      (is (broadcast-test (!randn `(1)) (!randn `(10))))
      (is (broadcast-test (!randn `(10)) (!randn `(10)))))

; Are they OK when batch is enabled?

(test broadcasting-blas-3d-repeats
      (is (broadcast-test (!randn `(10 10 1)) (!randn `(10 10 10))))
      (is (broadcast-test (!randn `(10 10 10)) (!randn `(10 10 1))))
      
      (is (broadcast-test (!randn `(10 1 10)) (!randn `(10 10 10))))
      (is (broadcast-test (!randn `(10 10 10)) (!randn `(10 1 10))))

      (is (broadcast-test (!randn `(10 10 1)) (!randn `(10 1 10))))
      (is (broadcast-test (!randn `(10 1 10)) (!randn `(10 10 1)))))

(test broadcasting-blas-3d
      (is (broadcast-test (!randn `(1 10 1)) (!randn `(10 10 10))))
      (is (broadcast-test (!randn `(10 10 10)) (!randn `(1 10 1))))
      
      (is (broadcast-test (!randn `(1 1 10)) (!randn `(10 10 10))))
      (is (broadcast-test (!randn `(10 10 10)) (!randn `(1 1 10))))

      (is (broadcast-test (!randn `(1 10 1)) (!randn `(10 1 10))))
      (is (broadcast-test (!randn `(10 1 10)) (!randn `(1 10 1)))))

(test broadcasting-blas-4d-repeats
      (is (broadcast-test (!randn `(10 1 10 1)) (!randn `(10 10 10 10))))
      (is (broadcast-test (!randn `(10 10 10 10)) (!randn `(10 1 10 1))))
      
      (is (broadcast-test (!randn `(10 1 1 10)) (!randn `(10 10 10 10))))
      (is (broadcast-test (!randn `(10 10 10 10)) (!randn `(10 1 1 10))))

      (is (broadcast-test (!randn `(10 1 10 1)) (!randn `(10 10 1 10))))
      (is (broadcast-test (!randn `(10 10 1 10)) (!randn `(10 1 10 1)))))

(test broadcasting-blas-4d
      (is (broadcast-test (!randn `(10 1 10 1)) (!randn `(1 10 10 10))))
      (is (broadcast-test (!randn `(1 10 10 10)) (!randn `(10 1 10 1))))
      
      (is (broadcast-test (!randn `(10 1 1 10)) (!randn `(1 10 10 10))))
      (is (broadcast-test (!randn `(1 10 10 10)) (!randn `(10 1 1 10))))

      (is (broadcast-test (!randn `(10 1 10 1)) (!randn `(1 10 1 10))))
      (is (broadcast-test (!randn `(1 10 1 10)) (!randn `(10 1 10 1)))))
