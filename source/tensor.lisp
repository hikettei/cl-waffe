
(in-package :cl-waffe)


(defparameter *print-char-max-len* 5)
(defparameter *print-arr-max-size* 6)
(defparameter *print-mat-max-size* 3)

(defstruct (WaffeTensor (:print-function
			 (lambda (tensor stream depth)
			   (declare (ignore depth))
			   (format stream (render-tensor tensor))))
	                (:constructor
			    tensor
			    (value &optional (backend :cpu)
			     &aux (data value) (backend backend) (grad `(nil nil))))
			(:constructor
			    const
			    (value &optional (backend :cpu)
			     &aux (data value) (backend backend) (grad nil))))
  data grad-tmp backward backend grad variables state)

(defun data (tensor)
  (slot-value tensor 'data))

(defun is-tensor (grad)
  (typep grad 'WaffeTensor))

(defun repeat (tensor n)
  (map 'list
       (lambda (x)
	 (declare (ignore x))
	 (const n))
       (slot-value tensor 'variables)))

(defmacro nth-var (tensor n)
  `(nth ,n (slot-value ,tensor 'variables)))

(defmacro nth-tensor (tensor n s)
  ; the nth variavle of tensor
  `(slot-value (nth-var ,tensor ,n) ,s))

(defun grad (tensor)
  (unless (slot-value tensor 'grad)
    (error "The tensor is not a parameter"))

  (if (typep (slot-value tensor 'grad) 'cons)
      (error "Before using grad, you need to call (backward tensor)"))

  (slot-value tensor 'grad))

(defmacro parameter (tensor)
  ; enable grad
  `(with-slots ((data data) (backend backend)) ,tensor
     (tensor data backend)))
  
(defun backward (tensor)
  (if (slot-value tensor 'backward)
      (let ((state (slot-value tensor 'state))
	    (grad  (if (slot-value tensor 'grad-tmp)
		       (slot-value tensor 'grad-tmp)
		       (repeat tensor 1))))
	(let ((grads (apply (slot-value tensor 'backward) state grad)))
	  (dotimes (i (length (slot-value tensor 'variables)))
	    (if (nth-tensor tensor i 'grad-tmp)
		(setf (nth-tensor tensor i 'grad-tmp)
		      (repeat (nth-var tensor i) (data (add (nth-tensor tensor i 'grad-tmp) (nth i grads)))))
		(setf (nth-tensor tensor i 'grad-tmp) (repeat (nth-var tensor i) (data (nth i grads)))))
	    (if (nth-tensor tensor i 'grad)
		(if (typep (nth-tensor tensor i 'grad) 'cons)
		    (setf (nth-tensor tensor i 'grad) (data (nth i grads)))
		    (setf (nth-tensor tensor i 'grad) (data (add (nth-tensor tensor i 'grad)
							    (data (nth i grads)))))))
	    (backward (nth-var tensor i)))))
      (setf (slot-value tensor 'grad-tmp) (if (slot-value tensor 'grad)
					  (repeat tensor 0)
					  (repeat tensor 0)))))

(defmacro numcl-to-waffe (waffe-name numcl-name)
  `(defmacro ,waffe-name (&rest args)
     `(const (,',numcl-name ,@args))))

(defmacro n2w (waffe-name numcl-name)
  `(numcl-to-waffe ,waffe-name ,numcl-name))

(n2w zeros numcl:zeros)
(n2w arange numcl:arange)

(defclass gaussiandb () ((mean :initform nil
			       :initarg :mean
			       :accessor gaussiandb-mean)
			 (var :initform nil
			      :initarg :var
			      :accessor gaussiandb-var)))

(defun double-random ()
  (let ((i (random 1.0)))
    (if (eq i 0.0)
	(setq i (double-random))) i))

(defmethod gaussiandb-random ((gs gaussiandb))
  (let* ((r (double-random))
	 (c (sqrt (* -2 (log r)))))
    (if (< (double-random) 0.5)
	(+    (* c
	      (sin (* 2.0 pi (double-random)))
	      (gaussiandb-var gs))
	      (gaussiandb-mean gs))
	(+    (* c
	      (cos (* 2.0 pi (double-random)))
	      (gaussiandb-var gs))
	      (gaussiandb-mean gs)))))

(defun getrandn ()
  (let ((u1 (loop for x = (random 1.0)
                  when (> x 0.0)
                    return x))
        (u2 (loop for x = (random 1.0)
                  when (> x 0.0)
                    return x)))
    (values (* (sqrt (* -2 (log u1))) (cos (* 2 pi u2)))
            (* (sqrt (* -2 (log u1))) (sin (* 2 pi u2))))))

(defun random-tensor (dims limit)
  (let* ((res (make-array dims))
         (upper-limit (if (listp limit) (second limit) limit))
         (lower-limit (if (listp limit) (first limit) 0))
         (len (if (listp dims) (reduce #'* dims) dims))
         (tmp-limit (- upper-limit lower-limit)))
    (loop for n from 0 to (1- len)
          do (setf (row-major-aref res n)
                   (+ (random tmp-limit) lower-limit)))
    (const (numcl:asarray res))))

(defun randn (&rest dims)
  (let* ((res (make-array dims))
         (len (if (listp dims) (reduce #'* dims) dims)))
    (loop for n from 0 to (1- len)
          do (setf (row-major-aref res n) (getrandn)))
    (const (numcl:asarray res))))

(defun normal (dims &optional (mean 2.0) (var 1.0))
  (let* ((gb (make-instance 'gaussiandb :mean mean :var var))
	 (res (make-array dims))
         (len (if (listp dims) (reduce #'* dims) dims)))
    (loop for n from 0 to (1- len)
          do (setf (row-major-aref res n) (gaussiandb-random gb)))
    (const (numcl:asarray res))))

(defun shape (tensor &optional (nth nil))
  (unless (typep (data tensor) 'array)
    ;error
    )
  
  (if nth
      (nth (data (assure-tensor nth)) (numcl:shape (data tensor)))
      (numcl:shape (data tensor))))

(defun astensor (arr)
  (const (numcl:asarray arr)))

(defun ones-like (arr)
  (const (numcl:ones-like (data arr))))

(defun write-description (res backward backend)
  (write-string (format nil " :device :~a :backward ~A" backend backward) res))

(defun reduce-str (obj)
  (let ((str (format nil "~a" obj)))
    (if (>= (length str) *print-char-max-len*)
	(concatenate 'string (subseq str 0 *print-char-max-len*) "...")
	str)))

(defun pprint-1d-vector (stream data)
  (if (> (length (numcl:shape data)) 1)
      (error ""))
  
  (if (>= (numcl:size data) *print-arr-max-size*)
      (write-string (format nil "(~A ~A ~2~ ~A ~A)"
			    (reduce-str (numcl:aref data 0))
			    (reduce-str (numcl:aref data 1))
			    (reduce-str (numcl:aref data (- (numcl:size data) 2)))
			    (reduce-str (numcl:aref data (1- (numcl:size data)))))
		    stream)
      (progn
	(write-string "(" stream)
	(dotimes (i (numcl:size data))
	  (write-string (format nil "~A" (reduce-str (numcl:aref data i))) stream)
	  (unless (= i (1- (numcl:size data)))
	    (write-string " " stream)))
	(write-string ")" stream))))

(defun pprint-vector (stream data &optional (newline T) (indent-size 0))
  (case (length (numcl:shape data))
    (1
     (pprint-1d-vector stream data))
    (T
     (write-string "(" stream)
     (if (< (car (numcl:shape data)) *print-mat-max-size*)
	 (progn
	   (dotimes (i (car (numcl:shape data)))
	     (pprint-vector stream (numcl:aref data i) newline (1+ indent-size))
	     (unless (= i (1- (car (numcl:shape data))))
	       (if newline
		   (progn
		     (write-char #\Newline stream)
		     (dotimes (k (1+ indent-size))
		       (write-string " " stream)))
		   (write-string " " stream))))
	   (write-string ")" stream))
	 (progn
	   (labels ((render-v (line do-newline)
		      (pprint-vector stream line newline (1+ indent-size))
		      (if do-newline
			  (if newline
			      (progn
				(write-char #\Newline stream)
				(if (= 2 (length (numcl:shape data)))
				    (progn
				      (dotimes (_ (+ (* 2 indent-size) 3))
					(declare (ignore _))
					(write-string " " stream))
				      (write-string "..." stream)
				      (write-char #\Newline stream)))
				(dotimes (k (1+ indent-size))
				  (write-string " " stream)))
			      (write-string " " stream)))))
	     (render-v (numcl:aref data 0) T)
	     ;(render-v (numcl:aref data 1) T)
	     
	     ;(render-v (numcl:aref data (- (car (numcl:shape data)) 2)) T)
	     (render-v (numcl:aref data (1- (car (numcl:shape data)))) NIL)
	     (write-string ")" stream)))))))

(defun render-tensor (tensor &optional (newline T) (indent-size 0))
  (with-slots ((contents data) (backward backward) (backend backend) (grad grad)) tensor
    (with-output-to-string (res)
      (if (null grad)
	  (write-string "#Const(" res)
	  (write-string "#Parameter{" res)) ;11
      
      (if (or (typep contents 'vector) (typep contents 'array))
	  (progn
	    (pprint-vector res contents newline (if (null grad)
						    (+ indent-size (length "#Const("))
						    (+ indent-size (length "#Parameter{")))) ; Numcl Array
	    (write-string (format nil " :shape ~a" (numcl:shape contents)) res)
	    (unless (null grad)
	      (write-description res backward backend))
	    (if (null grad)
		(write-string ")" res)
		(write-string "}" res)))
	  (progn ; Simple data
	    (write-string (format nil "~A" contents) res)
	    (unless (null grad)
	      (write-description res backward backend))
	    (if (null grad)
		(write-string ")" res)
		(write-string "}" res))))
      res)))
