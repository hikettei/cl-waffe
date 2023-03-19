
(in-package :cl-waffe-test)

(in-suite :test)

#|
Testing arithmetic operators
|#

(defparameter a (!add 10 (!randn `(10 10))))
(defparameter b (!add 10 (!randn `(10 10))))

(defparameter c (const 100.1))

(defparameter a-copy (const (mgl-mat:copy-mat (data a))))
(defparameter b-copy (const (mgl-mat:copy-mat (data b))))

(defun no-side-effects? (a1 b1)
  (and (M= (data a1) (data a-copy))
       (M= (data b1) (data b-copy))))

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
	   (getacc 5 r5 a a)
	   (no-side-effects? a b)))))

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
	   (getacc r2 b)
	   (no-side-effects? a b)))))

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
	   (getacc r2 ones)
	   (no-side-effects? a b)))))

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
	   (getacc r2 ones)
	   (no-side-effects? a b)))))

(defun pow-test ()
  (let* ((k (!randn `(10 10)))
	 (k-copy (const (mgl-mat:copy-mat (data k))))
	 (r (!pow k 2)))
    (and
     (mgl-mat:M= (value k) (value k-copy))
     (mgl-mat:M= (value r)
		 (mgl-mat:.expt! (value k) 2)))))

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

; Tests for squeeze and unsqueeze including !reshape
(defun squeeze-test ()
  (let* ((t1 (!randn `(10 10)))
	 (t1-copy (const (mgl-mat:copy-mat (data t1)))))
    (and (equal (!shape (!unsqueeze t1)) '(1 10 10))
	 (equal (!shape (!squeeze (!unsqueeze t1)))
		'(10 10))
	 (mgl-mat:M= (value t1) (value t1-copy)))))

(defun repeat-test ()
  (let* ((target-1 (const 1.0))
	 (target-2 (!randn `(1 10)))
	 (target-3 (!randn `(10 10 10 10 10)))
	 (t3-first (!aref target-3 t 0))
	 (t3-res (!repeats target-3 1 10)))
    (and (= (data (!sum (!repeats target-1 0 1000)))
	    1000.0)
	 (equal (!shape (!repeats target-2 0 10))
		`(10 10))
	 (not (find nil (loop for dim fixnum upfrom 0 below 100
			      collect (mgl-mat:M=
				       (data t3-first)
				       (data (!aref t3-res t dim)))))))))
(defun transpose1-test ()
  (let ((t1 (!randn `(1 10 12))))
    (equal (!shape (!transpose1 t1 2 1 0)) '(12 10 1))))

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

; labeling
(defun test-cross-entropy1 ()
  (let ((result t))
    (dotimes (i 10)
      (let ((loss (cl-waffe.nn:softmax-cross-entropy (!randn `(10 10 10))
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


(defun setfaref-backward ()
  (let ((tensor1 (parameter (!randn `(10 10 10))))
	(tensor2 (!add (!randn `(10 10)) 1.0))
	(tensor3 nil))
    (setq tensor3 (setf (!aref tensor1 0) tensor2)) ; kore kuso
    (backward (!sum tensor3))
    (grad tensor1)))

(defun aref-backward ()
  (let* ((tensor1 (parameter (!randn `(10 10 10))))
 	 (tensor2 (!add (!randn `(10 10)) (!aref tensor1 0))))
    (backward (!sum tensor2))
    (grad tensor1)))

(defun funcall-test ()
  (and (!log 1) (!add 1 1)))

(defun nd-matmul-test ()
  (let ((m3d (!randn `(10 10 10)))
	(m2d (!randn `(10 10))))
    (format t "~%Matmul: 3D * 2D~%")
    (time (assert (equal (!shape (!matmul m3d m2d))
			 '(10 10 10))
		  nil
		  "Matmul failed"))

    (format t "~%Matmul: 2D * 3D~%")
    (time (assert (equal (!shape (!matmul m2d m3d))
			 '(10 10 10))
		  nil
		  "Matmul Failed"))

    (format t "~%Matmul: 3D * 3D~%")
    (time (assert (equal (!shape (!matmul m3d m3d))
			 '(10 10 10))
		  nil
		  "Matmul Failed"))

    (let ((m2dt  (!transpose (!randn `(3 10))))
	  (m3dt1 (!transpose (!randn `(10 3 10))))
	  (m3dt  (!transpose (!randn `(10 10 3)))))
      (format t "~%Matmul: 2D.T * 3D~%")
      (time (assert (equal (!shape (!matmul (!transpose (!randn `(10 3))) m3d)) '(10 3 10))
		    nil
		    "Matmul test failed. 2D.T and 3D"))

      (format t "~%Matmul: 2D.T * 3D.T~%")
      (time (assert (equal (!shape (!matmul m2dt m3dt1))
			   '(10 10 3))
		    nil
		    "Matmul Test Failed"))

      (format t "~%Matmul: 3D.T * 2D~%")
      (time (assert (equal (!shape (!matmul m3dt m2d))
			   '(10 3 10))
		    (m3dt m2d)
		    "Matmul Test failed ~a.T and ~a"
		    m3dt
		    m2d))

      (format t "~%Matmul: 3D.T * 3D.T")
      (time (assert (equal (!shape (!matmul (!transpose (!randn `(10 70 50)))
					    (!transpose (!randn `(10 20 70)))))
			   '(10 50 20))
		    nil
		    "matmul test failed"))))
  t)

(defun argmax-test ()
  (let ((a (!randn `(10 10))))
    (mgl-mat:M=
     (mgl-mat:array-to-mat
      (max-position-column (mgl-mat:mat-to-array (data a))))
     (data (!argmax a)))))

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
      (is (nd-matmul-test))
      (is (test-cross-entropy))
      (is (test-cross-entropy1))
      (is (test-activations))
      (is (test-filter))
      ;(is (test-beta))
      (is (aref-backward))
      (is (funcall-test))
      (is (setfaref-backward))
      (is (argmax-test))
      )

