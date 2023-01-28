
(in-package :cl-waffe)


(defparameter *print-char-max-len* 5)
; todo comment...
(defparameter *print-arr-max-size* 6)
;
(defparameter *print-mat-max-size* 3)
;
(defparameter *default-backend* :mgl)

(defparameter mgl-mat:*DEFAULT-MAT-CTYPE* :float) ;double/float

(deftype WaffeSupportedDataType ()
  `(or fixnum float null cons function ratio)) ;cons?

(deftype WaffeDataType ()
  `(or fixnum
       float
       null
       cons
       function
       ratio))

(deftype waffe-array ()
  `(or mgl-mat:mat))

(defun waffe-array (c)
  (and (typep c 'simple-array)
       (every (lambda (e) (typep e 'waffesupporteddatatype)) c)))

(deftype WaffeTensorContentType ()
  `(or mgl-mat:mat
       simple-array
       waffesupporteddatatype))
					; (satisfies waffe-array)))

(eval-when (:compile-toplevel)
  (defstruct grad-tmp
    (value nil :type (or null waffetensor))
    (grad-called nil :type boolean)))

(defstruct (WaffeTensor (:print-function
			 (lambda (tensor stream depth)
			   (declare (ignore depth))
			   (format stream (render-tensor tensor))))
			(:constructor
			    sysconst
			    (value &key (backend *default-backend*)
				     (extend nil)
				     (is-node-tensor nil)
			     &aux (data (init-waffe-tensor-data value))
			       (backend (check-backend backend extend))
			       (calln 0)
			       (destructively-calln 0)
			       (grad nil)
			       (is-node-tensor is-node-tensor)
			       (is-mat (typep value 'mgl-mat:mat))
			       (destructive? t)
			       (grad-tmp (make-grad-tmp))))
	                (:constructor
			    tensor
			    (value &key (backend *default-backend*) (extend nil)
			     &aux (data (init-waffe-tensor-data value))
			       (is-ancestor-param t)
			       (calln 0)
			       (destructively-calln 0)
			       (is-mat (typep value 'mgl-mat:mat))
			       (is-param? t)
			       (backend (check-backend backend extend))
			       (grad `(nil nil))))
			(:constructor
			    const
			    (value &key (backend *default-backend*) (extend nil)
			     &aux (data (init-waffe-tensor-data value))
			       (backend (check-backend backend extend))
			       (calln 0)
			       (is-mat (typep value 'mgl-mat:mat))
			       (destructively-calln 0)
			       (grad nil)
			       (destructive? t))))
  (data nil :type waffetensorcontenttype)
  (grad-tmp (make-grad-tmp) :type grad-tmp)
  (backward nil :type boolean)
  (backend :mgl :type keyword)
  (grad nil :type waffetensorcontenttype)
  (variables nil :type list)
  state
  (is-mat nil :type boolean)
  (calln 0 :type fixnum) ; for debug, maybe unnecessary
  (is-param? nil :type boolean)
  (destructively-calln 0 :type fixnum) ; unnecessary
  (is-ancestor-param nil :type boolean)
  (is-next-destruct? nil :type boolean)
  (is-node-tensor nil :type boolean)
  (destructive? nil :type boolean) ; unnecessary
  (is-data-destructed? nil :type boolean))

(declaim (inline data))
(defun data (tensor)
  (declare (type waffetensor tensor))
  (waffetensor-data tensor))

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
	      var)))))

(eval-when (:compile-toplevel)
  (declaim (ftype (function (waffetensorcontenttype) (or waffetensorcontenttype)) init-waffe-tensor-data))
  ; Todo rewrite to macro, 処理系によってはエラー？
  (defun init-waffe-tensor-data (content)
  ; todo: coerce: simple-array -> mgl-mat
    (declare (type content waffedatatype))
    (etypecase content
      (ratio
	(if (eq mgl-mat:*default-mat-ctype* :double) ;...
	    (coerce content 'double-float)
	    (coerce content 'float)))
      (simple-array (mgl-mat:array-to-mat content))
      (T content))))

(defun check-backend (backend tensor)
  (if (null tensor)
      backend
      (waffetensor-backend tensor)))

(defmacro extend-from (new-tensor old-tensor)
  ; (extend-from (!randn `(10 10)) old-tensor) :backendとかを引き継ぐ
  (declare (ignore new-tensor old-tensor)))

(defmacro !allow-destruct (tensor)
  `(setf (waffetensor-is-next-destruct? ,tensor) t))

(defun (setf data) (val &optional tensor)
  (if tensor
      (setf (waffetensor-data tensor) val)
      tensor))

; is-tensor
(defun waffe-tensor-p (tensor)
  (typep tensor 'waffetensor))

(defmacro grad (tensor)
  `(progn
     (unless (typep ,tensor 'WaffeTensor)
       (error "The tensor is not a waffetensor"))
     
     (unless (waffetensor-grad ,tensor)
       (error "The tensor is not a parameter. Constants doesn't have a grad"))

     (if (typep (waffetensor-grad ,tensor) 'cons)
	 (error "A grad is nil. Please remain you need to call (backward out) before using a grad. When using ~%~a" ,tensor))

     (waffetensor-grad ,tensor)))

(defmacro parameter (tensor)
  ; make constants parameter
  `(with-slots ((data data) (backend backend)) ,tensor
     (tensor data :backend backend)))

(defun repeat-n (val n)
  (let ((a `(,val)))
    (dotimes (_ (1- n))
      (push val a))
    a))

(defun repeat-c (n &key (start 0))
  (let ((a `(,start))
	(i start))
    (dotimes (_ (1- n))
      (incf i 1)
      (push i a))
    (reverse a)))

(defmacro nth-var (tensor n)
  `(nth ,n (slot-value ,tensor 'variables)))

(defmacro nth-tensor (tensor n s)
  ; the nth variavle of tensor
  `(slot-value (nth-var ,tensor ,n) ,s))

(defmacro setfgradtmp (tensor value)
  `(progn
     (setf (grad-tmp-grad-called (waffetensor-grad-tmp ,tensor)) t)
     (if (and (typep (data ,tensor) 'mgl-mat:mat)
	      (typep (data ,value)  'mgl-mat:mat))
	 (if (equal (!shape ,tensor) (!shape ,value))
	     (setf (grad-tmp-value (waffetensor-grad-tmp ,tensor)) ,value)
	     (setf (grad-tmp-value (waffetensor-grad-tmp ,tensor)) (!reshape ,value (!shape ,tensor))))
	 (setf (grad-tmp-value (waffetensor-grad-tmp ,tensor)) ,value))))

(defun backward (tensor)
  (declare (type waffetensor tensor))
  (if (typep (data tensor) 'mgl-mat:mat)
      (unless (eq (!shape tensor) `(1))
	(error "grad can be implicitly created only for scalar outputs")))

  (setq *no-grad* t)
  (backward1 tensor)
  (setq *no-grad* nil)
  nil)

(declaim (inline step-next-node))

(defun step-next-node (tensor n)
  (if (waffetensor-is-ancestor-param (nth-var tensor n))
      (backward1 (nth-var tensor n))))

(declaim (ftype (function (waffetensor) null) backward1))
(defun backward1 (tensor)
  (declare (optimize (speed 3) (space 0) (safety 1))
	   (type waffetensor tensor))
  (cond
    ((waffetensor-backward tensor) ;Backward exists?
      (let* ((grad-tmp-before (waffetensor-grad-tmp tensor))
	     (grad-before (if (grad-tmp-grad-called grad-tmp-before) ;check if the node is a top
			      (grad-tmp-value grad-tmp-before)
			      (const 1))))
	(setf (waffetensor-is-node-tensor grad-before) t) ; assure grad-before won't be changed
	; calculating backward(state, dy) -> x.grad, y.grad...
        (progn
	  (let ((grads (funcall (the function (call-backward (waffetensor-state tensor))) grad-before)))
	    (declare (type list grads)) ; Print Error
	    (unless (= (length (waffetensor-variables tensor))
		       (length grads))
	      (error "backward error: The number of :forward args doesnt correspond with of :backward"))

	    (dotimes (n (length grads))
	      (setfgradtmp (nth-var tensor n) (nth n grads)))

	    (dotimes (n (length grads))
	      (step-next-node tensor n)))
	  nil)))
    (T
	(if (waffetensor-grad tensor) ; the tensor is the end of node.
	    (if (grad-tmp-value (waffetensor-grad-tmp tensor)) ; is grad-tmp already created?
		(if (typep (waffetensor-grad tensor) 'cons) ; is it first value? or not?
		    (let ((new-grad (grad-tmp-value (waffetensor-grad-tmp tensor))))
		      (if (typep (data new-grad) 'mgl-mat:mat)
			  (if (equal (!shape new-grad) (!shape tensor))
			      (setf (waffetensor-grad tensor) (data new-grad))
			      (setf (waffetensor-grad tensor) (data (!reshape new-grad (!shape tensor))))) ; is it due to bugs of reshape?
			  (setf (waffetensor-grad tensor) (data new-grad))))
		    (setf (waffetensor-grad tensor)
			  (data (!add (waffetensor-grad tensor)
				      (grad-tmp-value
				       (waffetensor-grad-tmp tensor))))))))))
  nil)


(defun !zeros (shape)
  (const (mgl-mat:make-mat shape :initial-element 0)))

(defun !ones (shape)
  (const (mgl-mat:make-mat shape :initial-element 1)))

(defun !fill (shape element)
  (const (mgl-mat:make-mat shape :initial-element element)))

(defmacro !arange (&rest args)
  `(const (mgl-mat:make-mat (numcl:shape (numcl:arange ,@args))
			    :initial-contents (numcl:arange ,@args))))
;backendsの引き継ぎは？
(defmacro !copy (tensor &aux (new-tensor (gensym)))
  `(let ((,new-tensor (!zeros-like ,tensor)))
     (mgl-mat:copy! (data ,tensor) (data ,new-tensor))
     ,new-tensor))

(declaim (ftype (function (waffetensor fixnum fixnum) waffetensor) !set-batch))
(defun !set-batch (dataset start-row-index batch-size)
  (declare (optimize (speed 3) (space 0) (safety 0))
	   (type waffetensor dataset)
	   (type fixnum start-row-index batch-size))
  (let ((dim (mgl-mat:mat-dimension (data dataset) 1)))
    (mgl-mat:reshape-and-displace! (data dataset)
                           (list batch-size dim)
                           (* start-row-index dim))
    dataset))

(defun !reset-batch (dataset)
  (let* ((dim (mgl-mat:mat-dimension (data dataset) 1))
         (len (/ (mgl-mat:mat-max-size (data dataset)) dim)))
    (reshape-and-displace! (data dataset) (list len dim) 0)
    dataset))

(defun !aref (tensor &rest dims) ; example: (aref vector 1 t t), (aref vector `(1 3) t t)
  (call (ArefTensor dims) tensor))

(defun !areflist (tensor dims)
  (call (ArefTensor dims) tensor))

(defun (setf !aref) (value tensor &rest dims)
  (setf tensor (setf (!areflist tensor dims) value)))

(defun (setf !areflist) (value tensor dims)
  ; For backward, you need to call it like (setq z (setf (!aref x ~) ~))
  ; To solve this problem, i guess i need more macros.
  (setf tensor (call (SetfArefTensor dims) tensor value)))
	 
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

(declaim (ftype (function (cons function) waffetensor) !random-with))
(defun !random-with (dims f)
  (declare (optimize (speed 3) (safety 0) (space 0))
	   (type cons dims)
	   (type function f))
  (let* ((res (make-array dims :initial-element 0))
         (len (if (listp dims) (reduce #'* dims) dims)))
    (loop for n from 0 to (1- len)
          do (setf (row-major-aref res n)
                   (funcall f)))
    (const res)))

(defun !normal (dims &optional (mean 2.0) (var 1.0))
  (let* ((res (!zeros dims))
         (len (if (listp dims) (reduce #'* dims) dims)))
    (loop for n from 0 to (1- len)
          do (setf (!row-major-aref res n) (gaussiandb-random var mean)))
    res))

(defmacro !randn (dims) ; this can be rewrited it by mgl-mat
  `(!normal ,dims 0 1))

(defun !beta (dims a b)
  (declare (ignore dims a b)))

(defun !gamma (dims scale)
  (declare (ignore dims scale)))

(defun !chisquare (dims df)
  (declare (ignore dims df)))

(declaim (ftype (function (cons single-float) waffetensor) !bernoulli))
(defun !bernoulli (dims rate)
  (declare (type cons dims)
	   (type single-float rate))
  (!modify (!zeros dims) :bernoulli (const rate)))

(defun !shape (tensor &optional (nth nil))
  (unless (typep (data tensor) 'waffe-array)
    (unless (typep (data tensor) 'function)
      (error "Fixnum/Double/Float doesn't have a shape")))
    
  (if nth
      (let* ((n (if (typep nth 'waffetensor)
	 	    (data nth)
		    nth))
	     (n (if (< n 0) (+ (!dims tensor) n) n)))
	(if (typep (data tensor) 'function)
	    (nth n (funcall (data tensor) t nil))
	    (mgl-mat:mat-dimension (data tensor) n)))
      (if (typep (data tensor) 'function)
	  (funcall (data tensor) t nil)
	  (mgl-mat:mat-dimensions (data tensor)))))

(defun !dims (tensor)
  (length (!shape tensor)))

(defun !size (tensor)
  (apply #'* (!shape tensor)))

(defun !size-1 (tensor)
  (1- (!size tensor)))

(defun !zeros-like (tensor)
  (!zeros (!shape tensor)))

(defun !ones-like (tensor)
  (!ones (!shape tensor)))

(defun !full-like ())

(defmacro detach (tensor)
  "Note: this macro doesn't clone data itself"
  `(const (data ,tensor)))

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

(defmacro !aref-array (array &rest args) ; possibly too slow...
  `(let ((res (data (!aref (const (mgl-mat:array-to-mat ,array)) ,@args))))
     (mgl-mat:mat-to-array (mgl-mat:reshape! res (cdr
						  (mgl-mat:mat-dimensions res)))))) ;unsqueeze

(defun !unsqueeze-array (array)
  (mgl-mat:mat-to-array (mgl-mat:reshape!
			 (mgl-mat:array-to-mat array) (cdr (array-dimensions array)))))

(defun pprint-1d-vector (stream data)
  (if (> (length (array-dimensions data)) 1)
      (error ""))
  (if (>= (apply #'* (array-dimensions data)) *print-arr-max-size*)
      (write-string (format nil "(~A ~A ~2~ ~A ~A)" ; todo: i wanna display last 3 digits.
			    (reduce-str (aref data 0))
			    (reduce-str (aref data 1))
			    (reduce-str (aref data (-  (length data) 2)))
			    (reduce-str (aref data (1- (length data)))))
		    stream)
      (progn
	(write-string "(" stream)
	(dotimes (i (apply #'* (array-dimensions data)))
	  (write-string (format nil "~A" (reduce-str (aref data i))) stream)
	  (unless (= i (1- (apply #'* (array-dimensions data))))
	    (write-string " " stream)))
	(write-string ")" stream))))

(defun pprint-vector (stream data &optional (newline T) (indent-size 0))
  (cond
    ((= 1 (length (array-dimensions data)))
     (pprint-1d-vector stream data))
    ((= 1 (car (array-dimensions data)))
     (write-string "(" stream)
     (pprint-vector stream (!unsqueeze-array data) newline (1+ indent-size))
     (write-string ")" stream))
    (T
     (write-string "(" stream)
     (if (< (car (array-dimensions data)) *print-mat-max-size*)
	 (let ((fd (car (array-dimensions data))))
	   (dotimes (i fd)
	     (pprint-vector stream (!aref-array data i) newline (1+ indent-size))
	     (unless (= i (1- fd))
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
				(dotimes (k (1+ indent-size))
				  (write-string " " stream))))))
	     (render-v (!aref-array data 0) T)
	     ;(render-v (numcl:aref data 1) T)
	     (if newline
		 (progn
		   (write-char #\newline stream)
		   (dotimes (_ (+ (* 2 indent-size) 3))
		     (write-string " " stream))
		   (write-string "..." stream)
		   (write-char #\newline stream)
		   (dotimes (k (1+ indent-size))
		     (write-string " " stream))))
	     ;(render-v (numcl:aref data (- (car (numcl:shape data)) 2)) T)
	     (render-v (!aref-array data (1- (car (array-dimensions data)))) NIL)
	     (write-string ")" stream)))))))

; ridiculously slow
(defun render-tensor (tensor &optional (newline T) (indent-size 0))
  (with-slots ((contents data) (backward backward) (backend backend) (grad grad)) tensor
    (with-output-to-string (res)
      (if (null grad)
	  (write-string "#Const(" res)
	  (write-string "#Parameter{" res))
      
      (if (or (typep contents 'array)
	      (typep contents 'vector))
	  (progn
	    (pprint-vector res contents newline (if (null grad)
						    (+ indent-size (length "#Const("))
						    (+ indent-size (length "#Parameter{"))))
	    (write-string (format nil " :mgl nil :shape ~a" (!shape contents)) res)
	    (unless (null grad)
	      (write-description res backward backend))
	    (if (null grad)
		(write-string ")" res)
		(write-string "}" res)))
	  (if (typep contents 'mgl-mat:mat)
	      (progn
		(pprint-vector res (mgl-mat:mat-to-array contents) newline
							 (if (null grad)
							     (+ indent-size (length "#Const("))
							     (+ indent-size (length "#Parameter{"))))
		(write-string (format nil " :mgl t :shape ~a" (mgl-mat:mat-dimensions contents)) res)
		(unless (null grad)
		  (write-description res backward backend))
		(if (null grad)
		    (write-string ")" res)
		    (write-string "}" res)))
	      (progn
		(write-string (format nil "~A" contents) res)
		(unless (null grad)
		  (write-description res backward backend))
		(if (null grad)
		    (write-string ")" res)
		    (write-string "}" res)))))
      res)))

