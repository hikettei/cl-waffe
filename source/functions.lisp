
(in-package :cl-waffe)

(defnode ReLUTensor nil
  :optimize t
  :parameters ((path-through nil) (zero-buff nil))
  :forward ((x) ; Todo rewrite more faster way.
	    (unless (self zero-buff)
	      (setf (self zero-buff) (!zeros (!shape x))))
	    (let ((mask (with-searching-calc-node :< x (self zero-buff))))
	      (save-for-backward path-through mask)
	      (!mul mask x)))
  :backward ((dy)
	     (list (!mul (self path-through) dy))))

(defun !relu (x)
  "Applying relu to x, return a new sysconst with making nodes.

Relu(x) = { 0 (x < 0), x (x > 0) }

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
  "Applyong sigmoid to x, return a new sysconst with making nodes.

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
  "Applying tanh to x, return a new sysconst with making nodes."
  (call (TanhTensor) (assure-tensor x)))

(defun !average (x)
  (let ((z (!sum x 1))
	(batch-size (!shape x 0)))
    (!div z batch-size)))

(defun !softmax-function (x &key (avoid-overflow t))
  "Applying softmax.

!softmax has three behaivour depending on the number of dimensions."
  (case (!dims x)
    (1 (!softmax-function (!unsqueeze x)))
    (2 (let* ((x1 (if avoid-overflow
		      (!sub x (!average x))
		      x))
	      (z (!sum (!exp x1) 1 t)))
	 (!div (!exp x1) z)))
    (3 (let* ((result (!zeros (!shape x)))) ; For batched inputs
	 (dotimes (i (!shape x 0))
	   (setq result (setf (!aref result i)
			      (!softmax-function (!squeeze (!aref x i) 0)))))
	 result))
    (T (error "!softmax: softmax only supports where (!dims tensor) <= 3."))))

(defmodel SoftMaxNode (avoid-overflow)
  :parameters ((avoid-overflow avoid-overflow))
  :forward ((x)
	    (!softmax-function x :avoid-overflow (self avoid-overflow))))

(defun !softmax (x &key (avoid-overflow t))
  (call (SoftMaxNode avoid-overflow) x))

; Todo :docstring
(defmodel model-list (model-args)
  :document (with-usage "model-list"
	      :overview "define model sequentially, (e.g. x = (sequence `((layer1) (layer2))), (call x 1 tensor) => layer1's output)"
	      :args "model1 model2 ..."
	      :forward "@cl:param(index) represents the index of models. @cl:param(args) is the arguments for index-th model."
	      :step-args "index &rest args")
  :parameters ((mlist model-args))
  :forward ((index &rest args)
	    (apply #'call (nth (data index) (self mlist)) args)))

(defnode ArefTensor (shape)
  :regard-as-node nil
  :parameters ((shape shape)
	       (base-shape T))

  :forward ((x) (setf (self base-shape) (!shape x))
		(apply #'!faref x (self shape))) ;thread-node??
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


(defun !faref (tensor &rest dims)
  (if (<= 2 (!dims tensor))
      (!faref-leg tensor dims)
      (!faref-2d tensor
		  nil
		  dims)))


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
  (!faref-2d tensor target dims))

; Error: dims isn't type of fixnum
(defun !faref-leg (tensor dims)
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
		     (incf total-bias (* (nth dim dims-result) bias))))))
    result))

(defun !faref-2d (tensor output dims)
  "(!dims tensor must be >= 2)"
  (declare (optimize (speed 3) (space 0))
	   (type waffetensor tensor)
	   (type list dims))
  (let* ((dims (cond ; assure (length dims) == (!dims tensor)
		 ((> (the fixnum (!dims tensor))
		     (the fixnum (length dims)))
		 (concatenate
		  'list
		  dims
		  (repeat-n t (the fixnum (- (the fixnum (!dims tensor))
					     (length dims))))))
		((= (the fixnum (!dims tensor)) (the fixnum (length dims)))
		 dims)
		(T
		 (error "!aref: dim ~a beyonds tensor's dim" dims))))
	 (dims-result (mapcar (lambda (x y)
				(the fixnum
				(typecase x
				  (fixnum 1)
				  (list
				   (unless (= (length x) 2)
				     (error "!aref: an argument is following: index, t, '(from start). ~a is invaild." x))
				   (the fixnum
					(- (the fixnum (second x))
					   (the fixnum (car x)))))
				  (T y))))
			      dims (!shape tensor)))
	 (dims-displacements (map 'list (lambda (x)
					  (the fixnum
					       (typecase x
						 (fixnum x)
						 (list (car x))
						 (T 0))))
				  dims)) ;where from copy starts?
	 (result
	   (if (null output)
	       (sysconst (make-mat dims-result :initial-element 0.0)
			 :thread-data (waffetensor-thread-data tensor))
	       output)))
    ; assure if dims are contigous
    (mapcar (lambda (x y)
	      (declare (type fixnum x))
	      (etypecase y
		(boolean nil)
		(fixnum (if (<= x y)
			    (error "!aref: the number ~A must be< ~a" y x)))
		(list (if (and (<= x (the fixnum (second y)))
			       (< (the fixnum (car y)) 0))
			  (error "!aref: the number ~a must be < ~a" y x)))
		(T (error "!faref: ~a is invaild argument" y))))		 
	    (if (null output)
		(!shape tensor)
		(!shape output))
	    dims)
    
    (labels ((%aref (nth
		     result-bias
		     tensor-bias
		     result-array
		     tensor-array)
	       (declare (optimize (speed 3) (space 0))
			(type fixnum nth result-bias tensor-bias)
			(type mat result-array tensor-array))
	       (let ((tensor-start-point (cl-waffe.backends.mgl:get-difference
					 tensor-array
					 nth))
		     (result-start-point (cl-waffe.backends.mgl:get-difference
					  result-array
					  nth))
		     (loop-iter (+ 
				   (the fixnum (nth nth dims-displacements))
				   (the fixnum (nth nth dims-result)))))
		 (declare (type fixnum tensor-start-point
				result-start-point))
		 (if (< (+ 2 nth) (the fixnum (!dims tensor)))
		     (loop for i fixnum
			   upfrom (nth nth dims-displacements) ; start points
			   below loop-iter
			   do 
			      (%aref (1+ nth)
				     (if (null output)
				     (the fixnum
					  (+ result-bias
					     (the fixnum
						  (*
						   (the fixnum
							(- i
							   (the fixnum
								(nth nth dims-displacements)))) ; result's bias of start index
						   result-start-point))))
				     (the fixnum
					  (+ result-bias
					     (the fixnum (* i result-start-point)))))
				     (+ tensor-bias
					(the fixnum (* i tensor-start-point)))
				     result-array
				     tensor-array)) ; the entry-point of (i-1)th dims
		     ; Copy For 2d Array

		     (let ((k (1+ nth)))
		       (cl-waffe.backends.mgl:copy-elements
			nth
			result-array
			tensor-array
			
			(the fixnum ; 1d
			   (- (+ (the fixnum (nth k dims-displacements))
				 (the fixnum (nth k dims-result)))
			      (the fixnum (nth k dims-displacements))))
			(the fixnum ; 2d
			   (- (+ (the fixnum (nth nth dims-displacements))
				 (the fixnum (nth nth dims-result)))
			      (the fixnum (nth nth dims-displacements))))
		      
			result-bias ; bias at current (2D)
			tensor-bias ; bias at current (2D)

			(if (null output)
			  0
			  (nth k dims-displacements))
			
			(if (null output)
			  (nth k dims-displacements)
			  0)

			(if (null output)
			  0
			  (nth nth dims-displacements))

			(if (null output)
			  (nth nth dims-displacements)
			  0)
			(if (null output)
			    1
			    0)))))
	       nil))
      
      (%aref 0 0 0 (data result) (data tensor))
      result)))
