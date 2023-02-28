
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

(defun register-to-benchmark (name cl-waffe-operation other-operations)
  (declare (type string name)
	   (type function cl-waffe-operation)
	   (type list other-operations))
  (push (benchmark name cl-waffe-operation other-operations) *benchmarks*)
  t)

(defun execute-benchmark (benchmark)
  (declare (type benchmarkset benchmark))
  
  nil)

(defmacro with-benchmark (name cl-waffe-benchmark))
