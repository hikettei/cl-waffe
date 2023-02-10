
(in-package :cl-waffe-test)

(in-suite :test)

(defparameter a (!add 10 (!randn `(10 10))))
(defparameter b (!add 10 (!randn `(10 10))))

(defparameter c (const 100.1))

(defun operate-test (fname lisp-func)
  (format t "Running Test of ~a~%" fname)
  (time (operate-test1 fname lisp-func)))

(defun operate-test1 (fname lisp-func)
  (let ((r1 (funcall fname a b))
	(r2 (funcall fname a c))
	(r3 (funcall fname c a))
	(r4 (funcall fname c b))
	(r5 (funcall fname a a))
	(c1 (!fill (!shape a) (data c))))
    (labels ((getacc (nth res x y)
	       (let ((result t))
		 (with-facets ((r  ((value res) 'backing-array :direction :input))
			       (x1 ((value x) 'backing-array))
			       (y1 ((value y) 'backing-array)))
		   (loop for i fixnum upfrom 0 below (!size x)
			 do (if (and
				 result
				 (~= (aref r i)
				    (funcall lisp-func (aref x1 i) (aref y1 i))))
				(setq result t)
				(setq result nil))))
		 (format t "~ath Test: ~a~%" nth result)
		 (unless result
		   (progn
		     (print x)
		     (print y)
		     (print res)))
		 result)))
      (and (getacc 1 r1 a b)
	   (getacc 2 r2 a c1)
	   (getacc 3 r3 c1 a)
	   (getacc 4 r4 c1 b)
	   (getacc 5 r5 a a)))))

(defun operate-func (fname lisp-func)
  (format t "Running Test of ~a~%" fname)
  (time (operate-func1 fname lisp-func)))

(defun operate-func1 (fname lisp-func)
  (let ((r1 (funcall fname a))
	(r2 (funcall fname b)))
    (labels ((getacc (res x)
	       (let ((result t))
		 (with-facets ((r  ((value res) 'backing-array :direction :input))
			       (x1 ((value x) 'backing-array)))
		   (loop for i fixnum upfrom 0 below (!size x)
			 do (if (and
				 result
				 (= (aref r i)
				    (funcall lisp-func (aref x1 i))))
				(setq result t)
				(setq result nil))))
		 result)))

      (and (getacc r1 a)
	   (getacc r2 b)))))

(defun test-softmax (x)
  (let ((r (!softmax x))
	(result t))
    (value r)
    (loop for i fixnum upfrom 0 below (!shape x 0)
	  do (progn
	       (let ((k (!sum x 1)))
		 (print k))))
    result))

(defun test-cross-entropy ()
  (let ((result t))
    (dotimes (i 100)
      (let ((loss (cl-waffe.nn:softmax-cross-entropy (!randn `(10 10))
						     (!ones `(10 10)))))
	(format t "CrossEntropyLoss:~a~%" loss)
	(if (and result (>= (data loss) 0))
	    (setq result t)
	    (setq result nil))))
    result))

(format t "Operating with Default Mode(cache=nil, jit=nil).~%")

(test cl-waffe-test
  (is (operate-test #'!add #'+))
  (is (operate-test #'!sub #'-))
  (is (operate-test #'!mul #'*))
  (is (operate-test #'!div #'/)) ;coerce single-float?

  (is (operate-func #'!exp #'exp))
  (is (operate-func #'!log #'log))
  (is (operate-func #'!sqrt #'sqrt))
  (is (operate-func #'!tanh #'tanh))
  (is (test-cross-entropy)))


