
(in-package :cl-waffe-test)

(in-suite :test)

(format t "Testing for broadcasting shapes~%")

(defparameter arg1 (!randn `(1000 1000)))
(defparameter arg2 (!randn `(1000 1)))


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
	 (result1 (time (!add m k))))
    (M= (value result) (value result1))))


(test broadcasting-test
      (is (simple-test1))
      (is (simple-test2))
      (is (simple-test3))
      (is (simple-test4))
      (is (simple-test5))
      (is (simple-test6)))
