
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
		       (geem! 1.0 (data x) (data y) 0.0 (data o)))))))

(with-benchmark "3D_Add"
  :cl-waffe (with-init-3d x y
	      (time (dotimes (i *loop-n*)
		      (!add x y))))
  :mgl-mat (with-init-3d x y
	     (time (dotimes (i *loop-n*)
		     (with-init-3d-out o
		       (axpy! 1.0 (data x) (data y))
		       (copy! (data y) (data o)))))))

(with-benchmark "3D_Sub"
  :cl-waffe (with-init-3d x y
	      (time (dotimes (i *loop-n*)
		      (!sub x y))))
  :mgl-mat (with-init-3d x y
	     (time (dotimes (i *loop-n*)
		     (with-init-3d-out o
		       (axpy! -1.0 (data x) (data y))
		       (copy! (data y) (data o)))))))

(with-benchmark "3D_Mul"
  :cl-waffe (with-init-3d x y
	      (time (dotimes (i *loop-n*)
		      (!mul x y))))
  :mgl-mat (with-init-3d x y
	     (time (dotimes (i *loop-n*)
		     (with-init-3d-out o
		       (geem! -1.0 (data x) (data y) 0.0 (data o)))))))


(with-benchmark "3D_Div"
  :cl-waffe (with-init-3d x y
	      (time (dotimes (i *loop-n*)
		      (!div x y))))
  :mgl-mat (with-init-3d x y
	     (time (dotimes (i *loop-n*)
		     (with-init-3d-out o
		       (.inv! (data y))
		       (geem! 1.0 (data x) (data y) 0.0 (data o)))))))

(with-benchmark "Broadcasting2DAdd"
  :cl-waffe (let ((x (!ones `(,*dim-n* ,*dim-n*)))
		  (y (!ones `(,*dim-n*))))
	      (time (dotimes (i *loop-n*)
		      (!add x y))))
  :mgl-mat (let ((x (data (!ones `(,*dim-n* ,*dim-n*))))
		 (y (data (!ones `(,*dim-n*)))))
	     (time (dotimes (i *loop-n*)
		     (let ((x1 (copy-mat x)))
		       ; scale-rows! is more faster way to compute bias?
		       (dotimes (k *dim-n*)
			 (reshape-and-displace! x1 `(,*dim-n*) k)
			 (axpy! 1.0 y x))
		       (reshape-and-displace! x1 `(,*dim-n* ,*dim-n*) 0))))))

(with-benchmark "Broadcasting2DSub"
  :cl-waffe (let ((x (!ones `(,*dim-n* ,*dim-n*)))
		  (y (!ones `(,*dim-n*))))
	      (time (dotimes (i *loop-n*)
		      (!sub x y))))
  :mgl-mat (let ((x (data (!ones `(,*dim-n* ,*dim-n*))))
		 (y (data (!ones `(,*dim-n*)))))
	     (time (dotimes (i *loop-n*)
		     (let ((x1 (copy-mat x)))
		       ; scale-rows! is more faster way to compute bias?
		       (dotimes (k *dim-n*)
			 (reshape-and-displace! x1 `(,*dim-n*) k)
			 (axpy! -1.0 y x))
		       (reshape-and-displace! x1 `(,*dim-n* ,*dim-n*) 0))))))

(with-benchmark "Broadcasting2DMul"
  :cl-waffe (let ((x (!ones `(,*dim-n* ,*dim-n*)))
		  (y (!ones `(,*dim-n*))))
	      (time (dotimes (i *loop-n*)
		      (!mul x y))))
  :mgl-mat (let ((x (data (!ones `(,*dim-n* ,*dim-n*))))
		 (y (data (!ones `(,*dim-n*)))))
	     (time (dotimes (i *loop-n*)
		     (let ((x1 (copy-mat x)))
		       (scale-rows! y x1))))))

(with-benchmark "Broadcasting2DDiv"
  :cl-waffe (let ((x (!ones `(,*dim-n* ,*dim-n*)))
		  (y (!ones `(,*dim-n*))))
	      (time (dotimes (i *loop-n*)
		      (!div x y))))
  :mgl-mat (let ((x (data (!ones `(,*dim-n* ,*dim-n*))))
		 (y (data (!ones `(,*dim-n*)))))
	     (time (dotimes (i *loop-n*)
		     (let ((x1 (copy-mat x))
			   (y1 (copy-mat y)))
		       (scale-rows! (.inv! y1) x1))))))

(with-benchmark "2D_1arg_Function_sin"
  :cl-waffe (with-init-2d-out x
	      (time (dotimes (i *loop-n*)
		      (!sin x))))
  :mgl-mat (time (dotimes (i *loop-n*)
		   (with-init-2d-out o
		     (.sin! (data o))))))

(with-benchmark "2D_1arg_Function_log"
  :cl-waffe (with-init-2d-out x
	      (time (dotimes (i *loop-n*)
		      (!log x))))
  :mgl-mat (time (dotimes (i *loop-n*)
		   (with-init-2d-out o
		     (.log! (data o))))))

(defun !!exp (x)
  (!allow-destruct x)
  (!exp x))

(defun !average1 (x)
  (declare (optimize (speed 3))
           (type waffetensor x))
  (let ((z (!sum x 1))
	(batch-size (!shape x 0)))
    (!!div z batch-size)))

(defun softmax1 (x)
  (declare (optimize (speed 3))
           (type waffetensor x))
  (let* ((x1 (!!mul -1.0 (!sub (!average1 x) x)))
         (xe (!!exp x1))
	 (z  (!sum xe 1)))
    (!!div xe z)))

(defun softmax2 (x)
  (declare (optimize (speed 3))
           (type waffetensor x))
  (let ((result (make-mat (!shape x)))
        (tmp    (make-mat `(1 ,@(cdr (!shape x)))))
	(x      (copy-mat (data x))))
       (sum! x tmp :axis 1)
       (scal! (/ (the fixnum (mat-dimension x 1))) tmp)
       (fill! 1.0 result)
       (scale-rows! tmp result)
       (axpy! -1.0 result x)
       (.exp! x)
       (sum! x tmp :axis 1)
       (fill! 1.0 result)
       (scale-rows! tmp result)
       (.inv! result)
    (const (.*! x result))))

(with-benchmark "2D_Softmax"
  :cl-waffe (with-init-2d-out x
	      (time (dotimes (i *loop-n*)
		      (softmax1 x))))
  :mgl-mat (time (dotimes (i *loop-n*)
		   (with-init-2d-out o
		     (softmax2 o)))))

(with-benchmark "2D_Copy"
  :cl-waffe (with-init-2d-out x
	      (time (dotimes (i *loop-n*)
		      (!aref x t))))
  :mgl-mat (time (dotimes (i *loop-n*)
		   (with-init-2d x y
		     (copy! (data x) (data y))))))


(defun start-benchmark (&key (dim-n 100) (loop-n 1000) (directory "./benchmark/benchmark.md") (speed-alert-min 1.5) (space-alert-min 1.5))
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

