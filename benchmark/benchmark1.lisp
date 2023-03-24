
(in-package :cl-waffe-benchmark)

; Here's benchmark compared to numpy.

(defparameter *N* 10000 "Trial N")
(defparameter *MATMUL_SIZE* `(16 32 64)); 128 256 512 1024 2048 4096 8192))

(defun mean (list)
  (/ (loop for i fixnum upfrom 0 below (length list)
	   sum (nth i list))
     (length list)))

; Counters

(defparameter *mm-try-i* 0)

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
		   (- t2 t1)))))
      (mean (loop for i fixnum upfrom 0 below *N*
		  collect (run-test))))))



(defparameter *MATMUL_SAVE_DIR* "./benchmark/results/matmul_waffe.png")

(defun compare-to-python ()
  (format t "LLA Config:~%~a~~%~%" cl-user::*lla-configuration*)
  (format t "ℹ️ Running matmul_2D...~%~%")

  (let ((result (loop for i fixnum upfrom 0 below (length *MATMUL_SIZE*)
		      collect (matmul_2d (nth i *MATMUL_SIZE*)))))
    (plot (map 'list #'(lambda (x) (coerce x 'double-float)) result)
      :x-seq *MATMUL_SIZE*
      :output *MATMUL_SAVE_DIR*
      :output-format :png)
    (format t "⭕️ The result is correctly saved at ~a~%" *MATMUL_SAVE_DIR*)))
