
(in-package :cl-waffe-benchmark)

; Here's benchmark compared to numpy.

"""
Todo: Display Configs/Environments, output to csv and integrate plots
memo:
export OPENBLAS_NUM_THREADS=1 (or 2? i guess there's no difference...)
"""

(defparameter *RESULT_DIR* "./benchmark/results/waffe_result.json")
(defparameter *RESULT_DIR_NUMPY* "./benchmark/results/numpy_result.json")

(defparameter *bench-results* nil)

(defparameter *N* 100 "Trial N")
(defparameter *backend-name* "OpenBLAS")

(defparameter *MATMUL_SIZE* `(16 32 64 256 512 1024 2048))

(defparameter *BROADCAST_SHAPE*
  `(((10 10 1) (1 10 10))
    ((100 100 1) (1 100 100))
    ((200 200 1) (1 200 200))
    ((300 300 1) (1 300 300))))

(defparameter *NN_SIZE* `(256 512 1024 2048)); 4096))

(defparameter *SLICE_SIZE* `(512 1024 2048 4096 8192))

(defparameter *BATCH_SIZE* 10000)

(defun mean (list)
  (/ (loop for i fixnum upfrom 0 below (length list)
	   sum (nth i list))
     (length list)))

; Counters

(defparameter *mm-try-i* 1)
(defparameter *broadcast-try-i* 1)
(defparameter *slice-try-i* 1)
(defparameter *nn-try-i* 1)

(defun matmul_2d (k)
  (format t "[~a/~a]  Testing on ~a*~a Matrix for ~a times~%"
	  *mm-try-i*
	  (length *MATMUL_SIZE*)
	  k
	  k
	  *N*)
  (incf *mm-try-i* 1)
  (let ((tensor (!randn `(,k ,k))))
    (labels ((run-test ()
	       (let ((t1 (get-internal-real-time)))
		 (!matmul tensor tensor)
		 (let ((t2 (get-internal-real-time)))
		   (/ (- t2 t1) internal-time-units-per-second)))))
      (mean (loop for i fixnum upfrom 0 below *N*
		  collect (run-test))))))

(defun broadcast-bench (shape)
  (format t "[~a/~a]  Testing on Matrix.size()[1]=~a for ~a times~%"
	  *broadcast-try-i*
	  (length *broadcast_shape*)
	  (second (car shape))
	  *N*)
  (incf *broadcast-try-i* 1)

  (let ((a (!randn (car shape)))
	(b (!randn (second shape))))
    (labels ((run-test ()
	       (let ((t1 (get-internal-real-time)))
		 (!add a b)
		 (let ((t2 (get-internal-real-time)))
		   (/ (- t2 t1) internal-time-units-per-second)))))
      (mean (loop for i fixnum upfrom 0 below *N*
		  collect (run-test))))))

(defun slice_2d (k)
  (format t "[~a/~a]  Testing on ~a*~a Matrix for ~a times~%"
	  *slice-try-i*
	  (length *SLICE_SIZE*)
	  k
	  k
	  *N*)
  (incf *slice-try-i* 1)
  (let ((tensor (!randn `(,k ,k))))
    (labels ((run-test ()
	       (let ((t1 (get-internal-real-time)))
		 (!aref tensor t '(200 400))
		 (let ((t2 (get-internal-real-time)))
		   (/ (- t2 t1) internal-time-units-per-second)))))
      (mean (loop for i fixnum upfrom 0 below *N*
		  collect (run-test))))))

(defun compute-nn (k)
  (format t "[~a/~a]  Testing on ~a*~a Matrix for ~a times~%"
	  *nn-try-i*
	  (length *NN_SIZE*)
	  *BATCH_SIZE*
	  k
	  *N*)
  (incf *nn-try-i* 1)
  (let ((tensor (!randn `(,*BATCH_SIZE* ,k)))
	(model  (cl-waffe.nn:denselayer k 10 t :relu)))
    (labels ((run-test ()
	       (let ((t1 (get-internal-real-time)))
		 (call model tensor)
		 (let ((t2 (get-internal-real-time)))
		   (/ (- t2 t1) internal-time-units-per-second)))))
      (mean (loop for i fixnum upfrom 0 below *N*
		  collect (run-test))))))

(defun append-result-json (name desc x-seq y-seq)
  (push `(:object-alist (,name . (:object-alist (:desc . ,desc)
						(:x-seq . ,x-seq)
						(:y-seq . ,y-seq))))
	*bench-results*))

(defun append-form-json (name desc)
  (push `(:object-alist (,name . ,desc)) *bench-results*))

(defun save-result-as-json ()
  (let ((result (reverse *bench-results*)))
    (with-open-file (stream *RESULT_DIR* :direction :output :if-exists :rename-and-delete :if-does-not-exist :create)
      (write-json result stream))))

(defparameter *MATMUL_SAVE_DIR* "./benchmark/results/matmul_waffe.png")
(defparameter *BROADCASTING_SAVE_DIR* "./benchmark/results/broadcasting_waffe.png")
(defparameter *SLICE_SAVE_DIR* "./benchmark/results/SLICE_waffe.png")
(defparameter *NN_SAVE_DIR* "./benchmark/results/dense_waffe.png")

(defun compare-to-python ()
  (with-no-grad
    (append-form-json "backend" (format nil "~a" cl-user::*lla-configuration*))
    (append-form-json "dtype" "single-float(i.e.: float32)")

    (format t "LLA Config:~%~a~%~%" cl-user::*lla-configuration*)
    (format t "ℹ️ Running matmul_2D...~%~%")

    (let ((result (loop for i fixnum upfrom 0 below (length *MATMUL_SIZE*)
			collect (matmul_2d (nth i *MATMUL_SIZE*)))))
      (plot (map 'list #'(lambda (x) (coerce x 'double-float)) result)
	:x-seq *MATMUL_SIZE*
	:title (format nil "matmul (cl-waffe + ~a) N=~a" *backend-name* *N*)
	:x-label "Matrix Size"
	:y-label "time (second)"
	:output *MATMUL_SAVE_DIR*
	:output-format :png)
      (append-result-json "matmul"
			  (format nil "~a, N=~a" *backend-name* *N*)
			  *MATMUL_SIZE*
			  result)
      (format t "⭕️ The result is correctly saved at ~a~%~%" *MATMUL_SAVE_DIR*))

    (format t "ℹ️ Running broadcasting...~%~%")
    (let ((result (loop for i fixnum upfrom 0 below (length *BROADCAST_SHAPE*)
			collect (broadcast-bench (nth i *BROADCAST_SHAPE*)))))

      (plot (map 'list #'(lambda (x) (coerce x 'double-float)) result)
	:title (format nil "broadcasting (cl-waffe + ~a) N=~a"
		       *backend-name*
		       *N*)
	:x-seq (map 'list #'(lambda (x) (second (car x))) *BROADCAST_SHAPE*)
	:x-label "Matrix Size"
	:y-label "time (second)"
	:output *BROADCASTING_SAVE_DIR*
	:output-format :png)

      (append-result-json "broadcasting"
			  (format nil "~a, N=~a" *backend-name* *N*)
			  (map 'list #'(lambda (x) (coerce x 'double-float)) result)
			  result)
      (format t "⭕️ The result is correctly saved at ~a~%~%" *BROADCASTING_SAVE_DIR*))

    (format t "ℹ️ Running slice_2D...~%~%")

    (let ((result (loop for i fixnum upfrom 0 below (length *SLICE_SIZE*)
			collect (slice_2d (nth i *SLICE_SIZE*)))))
      (plot (map 'list #'(lambda (x) (coerce x 'double-float)) result)
	:x-seq *SLICE_SIZE*
	:title (format nil "slicing (cl-waffe + ~a) N=~a" *backend-name* *N*)
	:x-label "Matrix Size"
	:y-label "time (second)"
	:output *SLICE_SAVE_DIR*
	:output-format :png)
      
      (append-result-json "slice"
			  (format nil "~a, N=~a" *backend-name* *N*)
			  *SLICE_SIZE*
			  result)
      (format t "⭕️ The result is correctly saved at ~a~%~%" *SLICE_SAVE_DIR*))


    (format t "ℹ️ Running Dense...~%~%")

    (let ((result (loop for i fixnum upfrom 0 below (length *NN_SIZE*)
			collect (compute-nn (nth i *NN_SIZE*)))))
      (plot (map 'list #'(lambda (x) (coerce x 'double-float)) result)
	:x-seq *NN_SIZE*
	:title (format nil "DenseLayer(ReLU) (cl-waffe + ~a) N=~a, BATCH_SIZE=~a" *backend-name* *N* *BATCH_SIZE*)
	:x-label "Matrix Size"
	:y-label "time (second)"
	:output *NN_SAVE_DIR*
	:output-format :png)

      (append-result-json "DenseLayer"
			  (format nil "~a, N=~a, BATCH_SIZE=~a" *backend-name* *N* *BATCH_SIZE*)
			  *NN_SIZE*
			  result)
      (format t "⭕️ The result is correctly saved at ~a~%~%" *NN_SAVE_DIR*))
    (save-result-as-json)
    (format t "✅ All benchmark are done and results are saved at ~a as a json file." *RESULT_DIR*)))

(defun load-result (path)
  (with-open-file (stream path :direction :input :if-does-not-exist :error)
    (read-json* :stream stream
		:float-format 'single-float
		:array-format :list)))

(defun generate-result (&key (result-md "./benchmark/Result.md"))
  (let ((waffe-result (load-result *RESULT_DIR*))
	(numpy-result (load-result *RESULT_DIR_NUMPY*)))
    (with-open-file (stream result-md
			    :direction :output
			    :if-exists :rename-and-delete
			    :if-does-not-exist :create)

      (format stream "# Benchmarking~%")
      (format stream "First, as a matrix arithmetic library, I measured benchmarks compared to NumPy as impartial as possible..~%~%")
      (format stream "Also, cl-waffe is also a deep learning framework. Benchmakrs compared to PyTorch is available.~%~%")


      (format stream "## Machine Environments~%~%")
      (format stream "|machine-type|machine-version|software-version|software-type|impl|~%")
      (format stream "|---|---|---|---|---|~%")
      (format stream "|~a|~a|~a|~a|~a|~%~%"
	      (machine-type)
	      (machine-version)
	      (software-version)
	      (software-type)
	      (format nil "~a [~a]"
		      (lisp-implementation-type)
		      (lisp-implementation-version)))

      (format stream "## cl-waffe and numpy~%~%")
      (format stream "")
      
      )))
