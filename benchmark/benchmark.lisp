
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
	      (time (dotimes (i *loop-n*)
		      (!sub x y))))
  :mgl-mat (with-init-2d x y
	     (time (dotimes (i *loop-n*)
		     (with-init-2d-out o
		       (axpy! -1.0 (data x) (data y))
		       (copy! (data y) (data o)))))))

(with-benchmark "2D_Mul"
  :cl-waffe (with-init-2d x y
	      (time (dotimes (i *loop-n*)
		      (!mul x y))))
  :mgl-mat (with-init-2d x y
	     (time (dotimes (i *loop-n*)
		     (with-init-2d-out o
		       (geem! -1.0 (data x) (data y) 0.0 (data o)))))))


(with-benchmark "2D_Div"
  :cl-waffe (with-init-2d x y
	      (time (dotimes (i *loop-n*)
		      (!div x y))))
  :mgl-mat (with-init-2d x y
	     (time (dotimes (i *loop-n*)
		     (with-init-2d-out o
		       (.inv! (data y))
		       (geem! -1.0 (data x) (data y) 0.0 (data o)))))))

(defun start-benchmark (&key (dim-n 1000) (loop-n 1000) (directory "./benchmark/benchmark.md") (speed-alert-min 1.5) (space-alert-min 1.5))
  (format t "✅ Benchmarking :cl-waffe~%")
  (format t "✅ The number of benchmarks is : ~a~%" (length *benchmarks*))
  
  (cl-cram:init-progress-bar bar "Benchmark" (length *benchmarks*))
  (fresh-line)
  (cl-cram:update bar 0)
  (fresh-line)
  (let ((*dim-n* dim-n)
	(*loop-n* loop-n)
	(*result*)
	(*space-alert-min* space-alert-min)
	(*speed-alert-min* speed-alert-min)
	(*benchmarks* (reverse *benchmarks*)))
    (dotimes (i (length *benchmarks*))
      (execute-benchmark (nth i *benchmarks*))
      (cl-cram:update bar 1)
      (fresh-line))
    (fresh-line)
    (format t "~%Collecting results...~%")
    (with-open-file (stream directory :direction :output :if-exists :supersede)
      (save-result stream))
    (format t "~%✅ Benchmarks are all done. The results are saved to ~a~%" directory)))

