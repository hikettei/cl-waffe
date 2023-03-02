
(in-package :cl-waffe-test)

(in-suite :test)

(format t "Testing for broadcasting shapes~%")

(defparameter arg1 (!randn `(1000 1000)))
(defparameter arg2 (!randn `(1000 1)))

(defun broadcast-1 (func a b)
  (cl-waffe.backends.mgl::broadcasting-apply-facet func a b))

(defun broadcast-2 (func a b)
  (cl-waffe.backends.mgl::broadcasting-apply-mgl func a b))

(defun broadcast-test (a b)
  (let ((a1 (broadcast-1 :+ a b))
	(b1 (broadcast-2 :+ a b))
	(a2 (broadcast-1 :- a b))
	(b2 (broadcast-2 :- a b))
	(a3 (broadcast-1 :* a b))
	(b3 (broadcast-2 :* a b)))
     (and (M= a1 b1)
	  (M= a2 b2)
	  (M= a3 b3))))
  
; most basic
(defun simple-test1 ()
  (format t "Test1: broadcasting (!add arg1 arg2)...")
  (let* ((arg2-r (!repeats arg2 1 1000))
	 (result (time (!add arg1 arg2)))
	 (result1 (time (!add arg1 arg2-r))))
    (M= (value result) (value result1))))

; Swap args
(defun simple-test2 ()
  (format t "Test2: broadcasting (!add arg2 arg1)...")
  (let* ((arg2-r (!repeats arg2 1 1000))
	 (result (time (!add arg2 arg1)))
	 (result1 (time (!add arg1 arg2-r))))
    (M= (value result) (value result1))))

; For other operators
(defun simple-test3 ()
  (format t "Test3: broadcasting (!sub arg2 arg1)...")
  (let* ((arg2-r (!repeats arg2 1 1000))
	 (result (time (!sub arg1 arg2)))
	 (result1 (time (!sub arg1 arg2-r))))
    (M= (value result) (value result1))))

; For other operators
(defun simple-test4 ()
  (format t "Test4: broadcasting (!mul arg2 arg1)...")
  (let* ((arg2-r (!repeats arg2 1 1000))
	 (result (time (!mul arg2 arg1)))
	 (result1 (time (!mul arg1 arg2-r))))
    (M= (value result) (value result1))))

; Testing unsqueeze
(defun simple-test5 ()
  (format t "Test5:")
  (let* ((k 1.0)
	 (k-tensor (!fill `(1) k))
	 (result  (time (!add arg1 k-tensor)))
	 (result1 (time (!add arg1 k))))
    (M= (value result) (value result1))))

; For 3D
(defun simple-test6 ()
  (format t "Test6:")
  (let* ((m (!randn `(100 100 1)))
	 (k (!repeats m 2 100))
	 (l (!randn `(100 100 100)))
	 (result  (time (!add m l)))
	 (result1 (time (!add k l))))
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
      (is (broadcast-test (!randn `(1)) (!randn `(10)))))

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

