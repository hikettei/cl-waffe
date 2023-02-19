
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

(defun operate-afunc (fname lisp-func)
  (format t "Running Test of ~a~%" fname)
  (time (operate-afunc1 fname lisp-func)))

(defun operate-afunc1 (fname lisp-func)
  (let* ((ones (!fill `(10 10) 0.10134))
	 (r1 (funcall fname ones))
 	 (r2 (funcall fname ones)))
    (labels ((getacc (res l)
	       (let ((result t))
		 (with-facets ((r  ((value res) 'backing-array :direction :input))
			       (x1 ((value l) 'backing-array)))
		   (loop for i fixnum upfrom 0 below 100
			 do (if (and
				 result
				 (= (aref r i)
				    (funcall lisp-func (aref x1 i))))
				(setq result t)
				(setq result nil))))
		 result)))

      (and (getacc r1 ones)
	   (getacc r2 ones)))))

(defun operate-afunc-cos (fname lisp-func)
  (format t "Running Test of ~a~%" fname)
  (time (operate-afunc1-cos fname lisp-func)))

(defun operate-afunc1-cos (fname lisp-func)
  (let* ((ones (!fill `(10 10) 1.0134))
	 (r1 (funcall fname ones))
 	 (r2 (funcall fname ones)))
    (labels ((getacc (res l)
	       (let ((result t))
		 (with-facets ((r  ((value res) 'backing-array :direction :input))
			       (x1 ((value l) 'backing-array)))
		   (loop for i fixnum upfrom 0 below 100
			 do (if (and
				 result
				 (= (aref r i)
				    (funcall lisp-func (aref x1 i))))
				(setq result t)
				(setq result nil))))
		 result)))

      (and (getacc r1 ones)
	   (getacc r2 ones)))))

(defun pow-test ()
  (let* ((k (!randn `(10 10)))
	 (r (!pow k 2)))
    (mgl-mat:M= (value r)
		(mgl-mat:.expt! (value k) 2))))

(defun dot-test ()
  (let* ((k (!randn `(10)))
	 (m (!randn `(10))))
    (= (data (!dot k m))
       (mgl-mat:dot (value k) (value m)))))

(defun matmul-test ()
  (let* ((k (!randn `(12 10)))
	 (m (!randn `(10 12)))
	 (l (!randn `(12 10))))
    (and (mgl-mat:M= (value (!matmul k m))
		     (mgl-mat:gemm! 1.0
				    (value k)
				    (value m)
				    0.0
				    (data (!zeros `(12 12)))))

	 (mgl-mat:M= (value (!matmul k (!transpose l)))
		     (mgl-mat:gemm! 1.0
				    (value k)
				    (value l)
				    0.0
				    (data (!zeros `(12 12)))
				    :transpose-b? t)))))

(defun sum-test ()
  (let ((k (!sum (!ones '(10 10)))))
    (= (data k) 100.0)))

(defun mean-test ()
  (let ((k (!mean (!ones '(10 10)))))
    (= (data k) 1.0)))

(defun squeeze-test ()
  (let ((t1 (!randn `(10 10))))
    (and (equal (!shape (!unsqueeze t1)) '(1 10 10))
	 (equal (!shape (!squeeze (!unsqueeze t1)))
		'(10 10)))))

(defun repeat-test ()
  (let ((t1 (!randn `(1 10))))
    (equal (!shape (!repeats t1 0 10))
	   '(10 10))))

(defun transpose1-test ()
  (let ((t1 (!randn `(1 10 12))))
    (equal (!shape (!transpose1 t1 2 1 0)) '(12 10 1))))

(defun !aref-test ()
  nil)

(defun !setf-aref-test ()
  nil)

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
    (dotimes (i 10)
      (let ((loss (cl-waffe.nn:softmax-cross-entropy (!randn `(10 10))
						     (!ones `(10 10)))))
	(when (= i 0)
	  (format t "CrossEntropyLoss:~a~%" loss))
	(if (and result (>= (data loss) 0))
	    (setq result t)
	    (setq result nil))))
    result))

(defun test-beta ()
  (let ((beta1 (!beta '(1000 1000) 5.0 5.0))
	(beta2 (!beta '(1000 1000) 0.5 0.5))
	(avg1 (/ 5.0 10.0))
	(avg2 (/ 0.5 1.0)))
    (if (and (~=1 (expt (data (!mean beta1)) 2) (expt avg1 2))
	     (~=1 (expt (data (!mean beta2)) 2) (expt avg2 2)))
	t
	nil)))

(defun test-filter ()
  (let ((x (!randn `(100 100))))
    (= 0.0 (data (!sum (!filter x #'(lambda (s) (declare (ignore s)) 0.0)))))))

(defun test-activations ()
  (let ((x (!randn `(10 10))))
    (and (!relu x)
	 (!tanh x)
	 (!sigmoid x)
	 (!leakey-relu x)
	 (!swish x)
	 (!gelu x)
	 (call (Swish :beta 1.0) x))))

(defun test-aref ()
  (let ((tensor1 (!reshape (!arange 0 100) '(10 10)))
	(tensor2 (!reshape (!arange 0 100) '(5 2 10))))

    (format t "Testing !aref...~%")

    (let ((tensor1-result (!sum (!aref tensor1 0 0))) ; 0
	  (tensor2-result (!sum (!aref tensor1 '(0 3) t))) ; 435.0
	  (tensor3-result (!sum (!aref tensor1 '(1 3) t))) ;390.0
	  (tensor4-result (!sum (!aref tensor1 '(1 3) '(2 4)))) ;70.0
	  (tensor5-result (!sum (!aref tensor2 '(1 3) t '(1 3))))
	  (tensor6 (!randn `(100 100 100))))

      (format t "Measuring performance of !aref... ~a~%" tensor6)
      (time (!aref tensor6 '(0 90) '(0 90) '(1 90)))
      
      (and (= (data tensor1-result) 0.0)
	   (= (data tensor2-result) 435.0)
	   (= (data tensor3-result) 390.0)
	   (= (data tensor4-result) 70.0)
	   (= (data tensor5-result) 292.0)))))

(defun test-setfaref ()
  (let ((tensor1 (!zeros `(10 10)))
	(tensor2 (!ones `(10 10 10))))
    (setf (!aref tensor2 '(0 1)) tensor1)
    (and (= (data (!sum (!aref tensor2 '(0 1)))) 0.0)
	 (> (data (!sum (!aref tensor2 '(2 -1)))) 0.0))))
	   

#|
(defun test-einsum ()
(let ((r1 (-> (!einsum (i j) (i j) -> (i j))
a
b)))
(and
(= (value r1) (value (!sum (!mul a b)))))))
|#

(format t "Operating with Default Mode(cache=nil, jit=nil).~%")

(test operator-test
      (is (operate-test #'!add #'+))
      (is (operate-test #'!sub #'-))
      (is (operate-test #'!mul #'*))
      (is (operate-test #'!div #'/)) ;coerce single-float?

      (is (pow-test))
      (is (dot-test))
      (is (matmul-test))
      (is (sum-test))
      (is (mean-test))
      (is (squeeze-test))
      (is (repeat-test))
      (is (transpose1-test))
      (is (operate-func #'!exp #'exp))
      (is (operate-func #'!log #'log))

      (is (operate-func #'!sqrt #'sqrt))
      
      (is (operate-func #'!sin #'sin))
      (is (operate-func #'!cos #'cos))
      (is (operate-func #'!tan #'tan))
      
      (is (operate-afunc #'!asin #'asin))
      (is (operate-afunc #'!acos #'acos))
      (is (operate-afunc #'!atan #'atan))
      
      (is (operate-func #'!sinh #'sinh))
      (is (operate-func #'!cosh #'cosh))
      (is (operate-func #'!tanh #'tanh))

      (is (operate-afunc #'!asinh #'asinh))
      (is (operate-afunc-cos #'!acosh #'acosh))
      (is (operate-afunc #'!atanh #'atanh))

      (is (test-aref))
      (is (test-setfaref))
      (is (test-cross-entropy))
      (is (test-activations))
      (is (test-filter))
      (is (test-beta))
     
      )

