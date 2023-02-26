
(in-package :cl-waffe-test)

(in-suite :test)

; tests for jit

(defparameter test-input (!randn `(10 10)))
(defparameter result (value (!softmax test-input)))

(defmacro with-jit (&body body)
  `(progn
     (setf cl-waffe.backends.mgl:*force-lazy-eval* t)
     (setf cl-waffe.backends.mgl:*verbose* t)

     (prog1
	 ,@body
       (setf cl-waffe.backends.mgl:*force-lazy-eval* nil)
       (setf cl-waffe.backends.mgl:*verbose* nil))))

(defun jittest1 () ; Tested on REPL ...
  (value (!softmax test-input)))

(test jit-test
      (is (jittest1)))

