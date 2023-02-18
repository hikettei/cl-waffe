
(in-package :cl-waffe-test)

(in-suite :test)

(format t "Testing for broadcasting shapes~%")

(defparameter arg1 (!randn `(1000 1000)))
(defparameter arg2 (!randn `(1000 1)))

(defparameter arg3 (!randn `(1000 1)))

(defun simple-test1 ()
  (let* ((arg2-r (!repeats arg2 1 1000))
	 (result (time (!add arg1 arg2)))
	 (result1 (time (!add arg1 arg2-r))))
    (M= (value result) (value result1))))
    
(test broadcasting-test
      (is (simple-test1)))
