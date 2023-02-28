
(in-package :cl-waffe-benchmark)

; Execute benchmarks

(defvar *dim-n*)

;(with-benchmark :Aadd ...)

(defun start-benchmark (&key (dim-n 100))
  (format t "✅ Benchmarking :cl-waffe~%")
  (format t "✅ The number of benchmarks is : ~a~%" (length *benchmarks*))
  
  (cl-cram:init-progress-bar bar "Benchmark" (length *benchmarks*))

  (let ((*dim-n* dim-n))
    (dotimes (i (length *benchmarks*))
      (execute-benchmark (nth i *benchmarks*))
      (cl-cram:update bar 1)))
  (fresh-line)
  (format t "✅ Benchmarks are all done. The results are saved to ~a~%" "A"))

