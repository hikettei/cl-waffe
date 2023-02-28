
(in-package :cl-waffe-benchmark)

; Here's a package for utils when benchmarking


(defparameter *benchmarks* nil)

(defstruct (BenchMarkSet
	    (:constructor benchmark
		(name cl-waffe-operation others
		 &aux (cl-waffe-operation cl-waffe-operation)
		   (other-operations others)
		   (name name))))
  "cl-waffe-operation must be given by lambda.
other-operation must be given by list which consist of (name lambda)
name must be given by string"
  (cl-waffe-operation #'(lambda () (error "Undefined")) :type function)
  (other-operations nil :type list)
  (name "No Name" :type string))


(defmacro with-benchmark (name &key cl-waffe (mgl-mat nil))
  `(register-to-benchmark ,name #'(lambda () ,cl-waffe)
			  ,(if (null mgl-mat)
			       nil
			       `(list "MGL-MAT" #'(lambda () ,mgl-mat)))))

(defun register-to-benchmark (name cl-waffe-operation other-operations)
  (declare (type string name)
	   (type function cl-waffe-operation)
	   (type list other-operations))
  (push (benchmark name cl-waffe-operation other-operations) *benchmarks*)
  t)

(defun execute-benchmark (benchmark)
  "Receiving benchmark structure, this function start benchmark following data with saving to specified file."
  (declare (type benchmarkset benchmark))
  (with-slots ((main-operation cl-waffe-operation)
	       (other-operations other-operations)
	       (main-name name))
      benchmark

    (funcall main-operation)

    (dolist (op other-operations)
      (funcall (second other-operations)))
    
    ; etc...
    ))

