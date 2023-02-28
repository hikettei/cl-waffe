
(in-package :cl-waffe-benchmark)

; Execute benchmarks

(with-benchmark "2D_Add"
  :cl-waffe (with-init-2d x y
	      (time (dotimes (i *loop-n*)
		      (!add x y))))
  :mgl-mat (with-init-2d x y
	     (time (dotimes (i *loop-n*)
		     (with-init-2d-out o
		       (axpy! 1.0 (data x) (data y))
		       (copy! (data y) (data o)))))))

(with-benchmark "2D_Sub"
  :cl-waffe (with-init-2d x y
	      (dotimes (i *loop-n*)
		(!sub x y))))

(defun start-benchmark (&key (dim-n 1000) (loop-n 1000) (directory "AAA"))
  (format t "✅ Benchmarking :cl-waffe~%")
  (format t "✅ The number of benchmarks is : ~a~%" (length *benchmarks*))
  
  (cl-cram:init-progress-bar bar "Benchmark" (length *benchmarks*))
  (fresh-line)
  (cl-cram:update bar 0)
  (fresh-line)
  (let ((*dim-n* dim-n)
	(*loop-n* loop-n)
	(*result*)
	(*benchmarks* (reverse *benchmarks*)))
    (dotimes (i (length *benchmarks*))
      (execute-benchmark (nth i *benchmarks*))
      (cl-cram:update bar 1)
      (fresh-line))
    (fresh-line)
    (save-result t)
    (format t "✅ Benchmarks are all done. The results are saved to ~a~%" directory)))

