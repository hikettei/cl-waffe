
(in-package :cl-waffe)


(defnode ReLUTensor nil
  :optimize t
  :parameters ((path-through nil) (zero-buff nil))
  :forward ((x)
	    (unless (self zero-buff)
		(setf (self zero-buff) (!zeros (!shape x))))
	    (let ((mask (with-searching-calc-node :< x (self zero-buff))))
	      (save-for-backward path-through mask)
	      (!mul mask x)))
  :backward ((dy)
	     (list (!mul (self path-through) dy))))

(defun !relu (x)
  "Calling relu with making node.
   Example: x' = { 0 (x < 0), x (x > 0)
   Input: x where x is waffe supported data type.
   Output: Tensor"
  (call (ReLUTensor) (assure-tensor x)))

(defnode SigmoidTensor nil
  :optimize t
  :parameters ((xi T))
  :forward ((x)
	    (save-for-backward xi x)
            (!div (!add 1 (!tanh (!div x 2))) (const 2)))
  :backward ((dy) (let ((p (!sigmoid (self xi))))
		    (list (!mul p (!mul dy (!sub 1 p)))))))

(defun !sigmoid (x)
  "Calling sigmoid with making node.
   Input: x where x is waffe supported data type.
   Output: Tensor"
  (call (SigmoidTensor) (assure-tensor x)))

(defnode TanhTensor nil
  :optimize t
  :parameters ((xi T))
  :forward ((x)
	    (save-for-backward xi x)
	    (with-searching-calc-node :tanh x))
  :backward ((dy)
	     (list (!mul dy (!sub (const 1) (!pow (!tanh (self xi)) 2))))))

(defun !tanh (x)
  (call (TanhTensor) (assure-tensor x)))

(defun !average (x)
  (let ((z (!sum x 1))
	(batch-size (!shape x 0)))
    (!div z batch-size)))

(defun !softmax (x &key (avoid-overflow t))
  (case (!dims x)
    (1 (!softmax (!unsqueeze x)))
    (2 (let* ((x1 (if avoid-overflow
		      (!sub x (!average x))
		      x))
	      (z (!sum (!exp x1) 1 t)))
	 (!div (!exp x1) z)))
    (3 (let* ((result (!zeros (!shape x)))) ; For batched inputs
	 (dotimes (i (!shape x 0))
	   (setq result (setf (!aref result i)
			      (!softmax (!squeeze (!aref x i) 0)))))
	 result))
    (T (error "!softmax: softmax only supports where (!dims tensor) <= 3."))))

; Todo :docstring
(defmodel model-list (model-args)
  ;Define model sequentially, (e.g. x = (sequence `((layer1) (layer2))), (call x 1 tensor) => layer1's output)
  :parameters ((mlist model-args))
  :forward ((index &rest args)
	    (apply #'call (nth (data index) (self mlist)) args)))

(defnode ArefTensor (shape)
  :regard-as-node nil
  :parameters ((shape shape)
	       (base-shape T))

  :forward ((x) (setf (self base-shape) (!shape x))
		(apply #'!faref x (self shape)))
  :backward ((dy)
	     (let ((dy-n (!zeros (self base-shape))))
	       (setf (!areflist dy-n (self shape)) dy)
	       (list dy-n))))

(defnode SetfArefTensor (shape)
  :parameters ((shape shape))
  :regard-as-node nil
  :forward ((x y)
	    (const (data (apply #'!write-faref x y (self shape)))))
  :backward ((dy)
	     (list dy (apply #'!faref dy (self shape)))))

; Error: dims isn't type of fixnum
(defun !faref (tensor &rest dims)
  "Example: (!aref vector 1 t t) (!aref vector '(1 3) t t)
  list args: (a b), cut arbitary dims in the range of a<=x<b"
  (let* ((dims (cond
		((> (!dims tensor) (length dims))
		 (concatenate ; complement lacking dims with t
		  'list
		  dims
		  (repeat-n t (- (!dims tensor) (length dims)))))
		((= (!dims tensor) (length dims))
		 dims)
		(T
		 (error "!aref: dim ~a beyonds tensor's dim" dims))))
	 (dims-result (mapcar (lambda (x y) (if (typep x 'fixnum)
						1
						(if (typep x 'list)
						    (progn
						      (unless (= (length x) 2)
							(error "!aref: an argument is following: index, t, '(from start). ~a is invaild." x))
						      (- (second x) (car x)))
						    y)))
			      dims (!shape tensor)))
	 (dims-displacements (map 'list (lambda (x)
					  (typecase x
					    (fixnum x)
					    (list (car x))
					    (T 0)))
				  dims))
	 (result (!zeros dims-result))
	 (bias 0)
	 (total-bias 0))

    (map 'list (lambda (x y)
		 (etypecase y
		   (boolean nil)
		   (fixnum (if (< x y)
			       (error "!aref: the number ~A must be<= ~a" y x)))
		   (list (if (< x (second y))
			     (error "!aref: the number ~a must be <= ~a" y x)))
		   (T nil)))			     
	 (!shape tensor) dims)

    (loop for dim upfrom 0 below (!dims tensor)
	  do
	     (if (or (= dim 0)
		     (not (eql T (nth dim dims))))
		 (progn
		 (dotimes (nth-axis (!shape result dim))
		   (setq bias
		    (cl-waffe.backends.mgl:write-to-nth-dim-with-range
		    (data result)
		    (data tensor)
		    dim
		    nth-axis
		    (nth dim dims-displacements)
		    total-bias)))
		 (if (not (eql T (nth dim dims)))
		     (incf total-bias (* (nth dim dims) bias))))))
    result))

(defun nth-bias (tensor aref-dims n)
  (let ((bias 0))
    (dotimes (i (1+ n))
      (incf bias (if (eql (nth i aref-dims) T)
		     0
		     (progn
		       ;(unless (< (!shape tensor i) (nth i aref-dims))
			; (error "(setf !aref): ~a is out of range for ~a. shape:~a" (nth i aref-dims) (!shape tensor i) (!shape tensor)))
		       (* (cl-waffe.backends.mgl:get-difference
			 (data tensor)
			 i)
			(nth i aref-dims))))))
    bias))

(defun !write-faref (target tensor &rest dims)
  "Example: (!aref vector 1 t t) (!aref vector '(1 3) t t)
  This can be called by (setf(!aref) ~)"
  (let* ((dims (cond
		((> (!dims tensor) (length dims))
		 (concatenate ; complement lacking dims with t
		  'list
		  dims
		  (repeat-n t (- (!dims tensor) (length dims)))))
		((= (!dims tensor) (length dims))
		 dims)
		(T
		 (error "!aref: dim ~a beyonds tensor's dim" dims))))
	 (result target))
    
    (unless (= (apply #'* (!shape (apply #'!aref target dims)))
	       (apply #'* (!shape tensor)))
      (error "(setf aref): Mismatch dims...(due to the waffe's bug)")) ; Todo: Cut tensor by args

    (map 'list (lambda (x y)
		 (etypecase y
		   (boolean nil)
		   (fixnum (if (< x y)
			       (error "!aref: the number ~a must be < ~a" y x)))
		   (list (if (< x (second y))
			     (error "!aref: the number ~a must be < ~a" y x)))
		   (T nil)))			     
	 (!shape target) dims)

    (loop for dim upfrom 0 below (length dims)
	  do (if (not (eql T (nth dim dims)))
		 (progn
		   (setf bias (nth-bias target dims dim))
		   (cl-waffe.backends.mgl:write-to-nth-dim-with-range1
		    (data result)
		    (data tensor)
		    (nth-bias target dims dim)))))
    result))
