
(in-package :cl-waffe)


(defparameter *print-char-max-len* 5)
; todo comment...
(defparameter *print-arr-max-size* 6)
;
(defparameter *print-mat-max-size* 3)
;
(defparameter *default-backend* :mgl)

; utils

(defun double-random ()
  (let ((i (random 1.0)))
    (if (eq i 0.0)
	(setq i (double-random)))
    i))

(defun gaussiandb-random (var mean)
  (let* ((r (double-random))
	 (c (sqrt (* -2 (log r)))))
    (if (< (double-random) 0.5)
	(+    (* c
	      (sin (* 2.0 pi (double-random)))
	      var)
	      mean)
	(+    (* c
	      (cos (* 2.0 pi (double-random)))
	      var)
	      mean))))

(defun repeat (tensor n)
  (map 'list
       (lambda (x)
	 (declare (ignore x))
	 (const n))
       (slot-value tensor 'variables)))

(defun repeat-n (val n)
  (let ((a `(,val)))
    (dotimes (_ (1- n))
      (push val a))
    a))

(defun repeat-c (n)
  (let ((a `(0))
	(i 0))
    (dotimes (_ (1- n))
      (incf i 1)
      (push i a))
    (reverse a)))

(defmacro nth-var (tensor n)
  `(nth ,n (slot-value ,tensor 'variables)))

(defmacro nth-tensor (tensor n s)
  ; the nth variavle of tensor
  `(slot-value (nth-var ,tensor ,n) ,s))



(deftype WaffeSupportedDataType ()
  `(or fixnum float))

(deftype Waffe-Array ()
  `(or mgl-mat:mat simple-array))

(defun waffe-array (c)
  (and (typep c 'simple-array)
       (every (lambda (e) (typep e 'WaffeSupportedDataType)) c)))

(deftype WaffeTensorContentType ()
  `(or mgl-mat:mat
       WaffeSupportedDataType))
      ; (satisfies waffe-array)))

(defun init-waffe-tensor-data (content)
  ; todo: coerce: simple-array -> mgl-mat

  (let* ((content (if (typep content 'WaffeTensor)
		      (data content)
		      content)))
    (unless (typep content 'WaffeTensorContentType)
      (error "WaffeTensor only supports of type of mgl-mat and fixnum/float but got: ~a" (type-of content)))

    content))

(defun check-backend (backend tensor)
  (if (null tensor)
      backend
      (waffetensor-backend tensor)))

(defstruct (WaffeTensor (:print-function
			 (lambda (tensor stream depth)
			   (declare (ignore depth))
			   (format stream (render-tensor tensor))))
	                (:constructor
			    tensor
			    (value &key (backend *default-backend*) (extend nil)
			     &aux (data (init-waffe-tensor-data value)) (backend (check-backend backend extend)) (grad `(nil nil))))
			(:constructor
			    const
			    (value &key (backend *default-backend*) (extend nil)
			     &aux (data (init-waffe-tensor-data value)) (backend (check-backend backend extend)) (grad nil))))
  data grad-tmp backward backend grad variables state)

(defmacro extend-from (new-tensor old-tensor)
  ; (extend-from (!randn `(10 10)) old-tensor) :backendとかを引き継ぐ
  (declare (ignore new-tensor old-tensor)))

(defmacro data (tensor)
  `(waffetensor-data ,tensor))

(defun (setf data) (val &optional tensor)
  (if tensor
      (setf (waffetensor-data tensor) val)
      tensor))

; is-tensor
(defun waffe-tensor-p (tensor)
  (typep tensor 'WaffeTensor))

(defmacro grad (tensor)
  `(progn
     (unless (waffetensor-grad ,tensor)
       (error "The tensor is not a parameter. Constants doesn't have a grad"))

     (if (typep (waffetensor-grad ,tensor) 'cons)
	 (error "A grad is nil. Please remain you need to call (backward out) before using a grad"))
  
  (waffetensor-grad ,tensor)))

(defmacro parameter (tensor)
  ; Make constants parameter
  `(with-slots ((data data) (backend backend)) ,tensor
     (tensor data :backend backend)))
  
(defun backward (tensor)
  (if (waffetensor-backward tensor)
      (let ((state (waffetensor-state tensor))
	    (grad  (if (waffetensor-grad-tmp tensor)
		       (if (typep (waffetensor-grad-tmp tensor) 'list)
			   (waffetensor-grad-tmp tensor)
			   (repeat tensor (waffetensor-grad-tmp tensor)))
		       (repeat tensor 1))))
	(dotimes (i (length (waffetensor-variables tensor)))
	  (let ((grads (apply (waffetensor-backward tensor) state (list (nth i grad)))))
	    (if (nth-tensor tensor i 'grad-tmp)
		(setf (nth-tensor tensor i 'grad-tmp)
		      (repeat (nth-var tensor i) (data (!add (nth-tensor tensor i 'grad-tmp) (nth i grads)))))
		(setf (nth-tensor tensor i 'grad-tmp) (repeat (nth-var tensor i) (data (nth i grads)))))
	    (if (nth-tensor tensor i 'grad)
		(if (typep (nth-tensor tensor i 'grad) 'cons)
		    (setf (nth-tensor tensor i 'grad) (data (nth i grads)))
		    (setf (nth-tensor tensor i 'grad) (data (!add (nth-tensor tensor i 'grad)
							         (nth i grads))))))
	    (backward (nth-var tensor i)))))
      (setf (slot-value tensor 'grad-tmp) (if (waffetensor-grad tensor)
					  (repeat tensor 0)
					  (repeat tensor 0)))))


(defmacro !zeros (shape &optional (dtype :double))
  `(const (mgl-mat:make-mat ,shape :ctype ,dtype :initial-element 0)))

(defmacro !ones (shape &optional (dtype :double))
  `(const (mgl-mat:make-mat ,shape :ctype ,dtype :initial-element 1)))

(defmacro !fill (shape element &optional (dtype  :double))
  `(const (mgl-mat:make-mat ,shape :ctype ,dtype :initial-element ,element)))

(defmacro !arange (&rest args)
  `(const (mgl-mat:make-mat (numcl:shape (numcl:arange ,@args))
			    :initial-contents (numcl:arange ,@args))))
;backendsの引き継ぎは？
(defmacro !copy (tensor &aux (new-tensor (gensym)))
  `(let ((,new-tensor (!zeros-like ,tensor)))
     (mgl-mat:copy! (data ,tensor) (data ,new-tensor))
     ,new-tensor))

(defnode CutTensor (result)
  :parameters ((result1 result))
  :forward ((x) (self result1))
  :backward ((dy) (list dy))) ; todo


(defun !aref (tensor &rest dims) ; example: (aref vector 1 t t)
  (let* ((tensor-dims (!shape tensor))
	 (dims (cond
		   ((> (!dims tensor) (length dims)) ;supply dims
		    (concatenate 'list dims (repeat-n t (- (!dims tensor) (length dims)))) )
		 ((= (!dims tensor) (length dims)) dims)
		 (T (error "!aref: dim ~a beyonds tensor's dim" dims))))
	 (dims-result (mapcar (lambda (x y) (if (typep x 'fixnum)
						 1
						 y))
			      dims tensor-dims))
	 (result (!zeros dims-result))
	 (dims-indices (mapcar (lambda (x y)
				 (if (typep x 'fixnum)
				     1
				     (repeat-c y)))
			       dims dims-result)))
    (labels ((next-node (drest args rargs)
	       (if (= (length args) (length dims))
		   (progn ; emmm...
		     (eval `(setf (mgl-mat:mref (data ,result) ,@rargs)
				  (mgl-mat:mref (data ,tensor) ,@args)))))
	       (if (typep (car drest) 'fixnum)
		   (next-node (cdr drest) (concatenate 'list args
						       `(,(car drest)))
			      (concatenate 'list rargs `(0)))
		   (dolist (m (car drest))
		     (next-node (cdr drest)
				(concatenate 'list args `(,m))
			        (concatenate 'list rargs `(,m)))))))

	(next-node dims-indices nil nil)
	(call (CutTensor result) tensor))))

(defun (setf !aref) (value &optional tensor &rest dims) ; (setf tensor value)
  (let* ((tensor-dims (!shape value))
	 (dims (cond
		   ((> (!dims tensor) (length dims)) ;supply dims
		    (concatenate 'list dims (repeat-n t (- (!dims tensor) (length dims)))) )
		 ((= (!dims tensor) (length dims)) dims)
		 (T (error "!aref: dim ~a beyonds tensor's dim" dims))))
	 (dims-result (mapcar (lambda (x y) (if (typep x 'fixnum)
						 1
						 y))
			      dims tensor-dims))
	 (result (!copy tensor))
	 (dims-indices (mapcar (lambda (x y)
				 (if (typep x 'fixnum)
				     1
				     (repeat-c y)))
			       dims dims-result)))
    (unless (and (mapcar (lambda (x y)
			   (if (typep y 'fixnum)
			       (eq x y)
			       t))
			 (!shape value) dims-result))
      (error "(setf !aref): mismatch dims ~a and ~a" (!shape value) dims-result))
    
    (labels ((next-node (drest args rargs)
	       (if (= (length args) (length dims))
		   (progn
		     (print args)
		     (print rargs)
		     (print result) (print value)
		     (eval `(setf (mgl-mat:mref (data ,result) ,@args)
				  (mgl-mat:mref (data ,value)  ,@rargs)))))
	       (if (typep (car drest) 'fixnum)
		   (next-node (cdr drest) (concatenate 'list args
						       `(,(car drest)))
			      (concatenate 'list rargs `(0)))
		   (dolist (m (car drest))
		     (next-node (cdr drest)
				(concatenate 'list args `(,m))
				(concatenate 'list rargs `(,m)))))))
      (next-node dims-indices nil nil)
      (setf tensor (call (CutBackward value) tensor)))))

(defmacro !where ()) ; todo
(defmacro !index ()) ; todo
  
(defmacro !row-major-aref (tensor index)
  `(mgl-mat:row-major-mref (data ,tensor) ,index))

(defmacro !with-mgl-operation (tensor var &body body)
  `(let ((,var (data ,tensor)))
     ,@body))

(defun !random (dims limit)
  ; if limit=fixnum, !random=randint
  ; if limit=float,  !random=random
  (let* ((res (!zeros dims))
         (upper-limit (if (listp limit) (second limit) limit))
         (lower-limit (if (listp limit) (first limit) 0))
         (len (if (listp dims) (reduce #'* dims) dims))
         (tmp-limit (- upper-limit lower-limit)))
    (loop for n from 0 to (1- len)
          do (setf (!row-major-aref res n)
                   (+ (random tmp-limit) lower-limit)))
    res))

(defun !random-with (dims f)
  (let* ((res (!zeros dims))
         (len (if (listp dims) (reduce #'* dims) dims)))
    (loop for n from 0 to (1- len)
          do (setf (!row-major-aref res n)
                   (funcall f)))
    f))

(defun !normal (dims &optional (mean 2.0) (var 1.0))
  (let* ((res (!zeros dims))
         (len (if (listp dims) (reduce #'* dims) dims)))
    (loop for n from 0 to (1- len)
          do (setf (!row-major-aref res n) (gaussiandb-random var mean)))
    res))

(defmacro !randn (dims)
  `(!normal ,dims 0 1))

(defun !binomial (dims n p)
  (declare (ignore dims n p)))

(defun !beta (dims a b)
  (declare (ignore dims a b)))

(defun !gamma (dims scale)
  (declare (ignore dims scale)))

(defun !chisquare (dims df)
  (declare (ignore dims df)))


(defun !shape (tensor &optional (nth nil))
  (unless (typep (data tensor) 'waffe-array)
    (error "Fixnum/Double/Float doesn't have a shape"))
  
  (if nth
      (mgl-mat:mat-dimension (data tensor) nth)
      (mgl-mat:mat-dimensions (data tensor))))

(defmacro !dims (tensor)
  `(length (!shape ,tensor)))

(defmacro !size (tensor)
  `(apply #'* (!shape ,tensor)))

(defmacro !size-1 (tensor)
  `(1- (!size ,tensor)))

(defmacro !zeros-like (tensor)
  `(!zeros (!shape ,tensor)))

(defmacro !ones-like (tensor)
  `(!ones (!shape ,tensor)))

(defmacro !full-like ())


(defun write-description (res backward backend)
  ; Parameter { ... <= here }
  (write-string (format nil " :device :~a :backward ~A" backend backward) res))

(defun reduce-str (obj)
  ; align string content of tensor following *print-char-max-len*
  ; Todo: 1.00000000001d0 <- 末端も表示
  (let ((str (format nil "~a" obj)))
    (if (>= (length str) *print-char-max-len*)
	(concatenate 'string (subseq str 0 *print-char-max-len*) "...")
	str)))


(defun pprint-1d-vector (stream data)
  (if (> (!dims data) 1)
      (error ""))
  (print data)
  (print (!aref data 0))
  (if (>= (!size data) *print-arr-max-size*)
      (write-string (format nil "(~A ~A ~2~ ~A ~A)"
			    (reduce-str (!aref data 0))
			    (reduce-str (!aref data 1))
			    (reduce-str (!aref data (-  (!size data) 2)))
			    (reduce-str (!aref data (1- (!size data)))))
		    stream)
      (progn
	(write-string "(" stream)
	(dotimes (i (!size data))
	  (write-string (format nil "~A" (reduce-str (!aref data i))) stream)
	  (unless (= i (!size-1 data))
	    (write-string " " stream)))
	(write-string ")" stream))))

(defun pprint-vector (stream data &optional (newline T) (indent-size 0))
  (case (!dims data)
    (1
     (pprint-1d-vector stream data))
    (T
     (write-string "(" stream)
     (if (< (!shape data 0) *print-mat-max-size*)
	 (progn
	   (dotimes (i (!shape data 0))
	     (pprint-vector stream (!aref data i) newline (1+ indent-size))
	     (unless (= i (1- (!shape data 0)))
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
				(if (= 2 (!dims data))
				    (progn
				      (dotimes (_ (+ (* 2 indent-size) 3))
					(write-string " " stream))
				      (write-string "..." stream)
				      (write-char #\Newline stream)))
				(dotimes (k (1+ indent-size))
				  (write-string " " stream)))
			      (write-string " " stream)))))
	     (render-v (!aref data 0) T)
	     ;(render-v (numcl:aref data 1) T)
	     
	     ;(render-v (numcl:aref data (- (car (numcl:shape data)) 2)) T)
	     (render-v (!aref data (1- (!shape data 0))) NIL)
	     (write-string ")" stream)))))))

(defun render-tensor (tensor &optional (newline T) (indent-size 0))
  (with-slots ((contents data) (backward backward) (backend backend) (grad grad)) tensor
    (with-output-to-string (res)
      (if (null grad)
	  (write-string "#Const(" res)
	  (write-string "#Parameter{" res))
      
      (if (or (typep contents 'array)
	      (typep contents 'vector)) ; typep tensor 'waffe-tensor
	  ; squeezeeが実装できるまでTensorでは使えない
          ; (!aref)するときに次数が下がらないため
          ; mgl -> numclにして表示する？ redefiningを直す
	  (progn
	    (pprint-vector res contents newline (if (null grad)
						    (+ indent-size (length "#Const("))
						    (+ indent-size (length "#Parameter{"))))
	    (write-string (format nil " :shape ~a" (!shape contents)) res)
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

