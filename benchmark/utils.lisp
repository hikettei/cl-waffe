
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

(defvar *dim-n*)
(defvar *loop-n*)
(defvar *result*)

(defmacro with-init-2d-out (o1  &body body)
  `(let ((,o1 (!ones `(,*dim-n* ,*dim-n*))))
     ,@body))

(defmacro with-init-3d-out (o1  &body body)
  `(let ((,o1 (!ones `(,*dim-n* ,*dim-n* ,*dim-n*))))
     ,@body))

(defmacro with-init-2d (x1 y1 &body body)
  `(let ((,x1 (!ones `(,*dim-n* ,*dim-n*)))
	 (,y1 (!ones `(,*dim-n* ,*dim-n*))))
     ,@body))

(defmacro with-init-3d (x1 y1 &body body)
  `(let ((,x1 (!ones `(,*dim-n* ,*dim-n* ,*dim-n*)))
	 (,y1 (!ones `(,*dim-n* ,*dim-n* ,*dim-n*))))
     ,@body))

(defmacro with-benchmark (name &key cl-waffe (mgl-mat nil))
  `(register-to-benchmark ,name #'(lambda () ,cl-waffe)
			  ,(if (null mgl-mat)
			       nil
			       `(list
				 (cons "MGL-MAT" #'(lambda () ,mgl-mat))))))

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

    (format t "~%🗒----Executing: ~a~%" main-name)
    (format t "--------~a:" "cl-waffe")
    (with-output-to-string (*trace-output*)
      ; handling
      (funcall main-operation))

    (format t " OK~%")
    
    (dolist (op other-operations)
      (format t "--------~a:" (car op))
      (with-output-to-string (*trace-output*)
	(funcall (second op)))

      (format t " OK~%"))
    
    ; etc...
    ))

