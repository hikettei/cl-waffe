
(in-package :cl-waffe)

(defgeneric assure-tensor (x))

(defmethod assure-tensor ((x waffetensor)) x)
(defmethod assure-tensor ((x fixnum))   (const x))
(defmethod assure-tensor ((x float))    (const x))
(defmethod assure-tensor ((x null))     (const x))
(defmethod assure-tensor ((x cons))     (const x))
(defmethod assure-tensor ((x function)) (const x))
(defmethod assure-tensor ((x ratio))    (const x))
(defmethod assure-tensor ((x mgl-mat:mat)) (const x))


(declaim (inline !add !sub !mul !div !exp !log !pow !sqrt !reshape !transpose !matmul))

(defnode AddTensor nil
  :parameters nil
  :forward  ((x y)
	     (with-searching-calc-node :add x y))
  :backward ((dy) (list dy dy)))

(defnode SubTensor nil
  :parameters ()
  :forward ((x y) (with-searching-calc-node :sub x y))
  :backward ((dy) (list dy (!mul dy (const -1)))))

(defnode MulTensor nil
  :parameters ((xi T) (yi T))
  :forward ((x y)
	    (setf (self xi) (data x))
	    (setf (self yi) (data y))
	    (with-searching-calc-node :mul x y))
  :backward ((dy) (list (!mul dy (self yi))
			(!mul dy (self xi)))))

(defnode DivTensor nil
  :parameters ((xi T) (yi T))
  :forward ((x y)
            (setf (self xi) (data x))
	    (setf (self yi) (data y))
	    (with-searching-calc-node :div x y))
  :backward ((dy) (list (!div dy (self yi))
			(!div (!mul (!mul dy (self xi)) -1)
			      (!pow (self yi) 2)))))

(defnode PowTensor nil
  :parameters ((xi T) (yi T))
  :forward ((x1 y1) (setf (self xi) (data x1))
		    (setf (self yi) (data y1))
		    (with-searching-calc-node :pow x1 y1))
  :backward ((dy)
	     (list (!mul (!mul dy (self yi)) (!pow (self xi) (!sub (self yi) 1)))
		   (!mul dy (!mul
			     (!log (self xi))
			     (!pow (self xi) (self yi)))))))

(defnode SqrtTensor nil
  :parameters ((xi T))
  :forward ((x1) (setf (self xi) x1)
		 (with-searching-calc-node :sqrt x1))
  :backward ((dy)
	     (list (!div dy (!mul 2 (!sqrt (self xi)))))))

(defnode LogTensor nil
  :parameters ((x1 T))
  :forward ((x1) (setf (self x1) x1)
		 (with-searching-calc-node :log x1))
  :backward ((dy) (list (!div dy (self x1)))))

(defnode ReshapeTensor (shape)
  :parameters ((prev-shape T) (shape shape))
  :forward ((x) (setf (self prev-shape) (assure-tensor (!shape x)))
		(with-searching-calc-node :reshape x (self shape)))
  :backward ((dy)
	     (list (!reshape dy (self prev-shape)))))

(defnode DotProductTensor nil
  :parameters ((xi T) (yi T))
  :forward ((x1 x2) ; only supports 2d and 2d arrays
		    (setf (self xi) x1)
		    (setf (self yi) x2)
		    (with-searching-calc-node :dot x1 x2))
  :backward ((dy)
	       (list (!dot dy (!transpose (self yi)))
		     (!dot (!transpose (self xi)) dy))))

(defnode TransposeTensor (shape)
  :parameters ((prev-shape T) (shape shape))
  :forward ((x)
	    (setf (self prev-shape) (!shape x))
	    (with-searching-calc-node :transpose x (self shape)))
  :backward ((d1) (!transpose d1 (self prev-shape))))

(defnode MeanTensor (axis)
  :parameters ((axis axis) (repeats T))
  :forward ((x)
	    (setf (self repeats) (assure-tensor (!shape x (self axis))))
	    (with-searching-calc-node :mean x (self axis)))
  :backward ((dy) (list (!repeats dy (self axis) (self repeats)))))

(defnode SumTensor (axis keepdims)
  :parameters ((axis axis) (repeats T))
  :forward ((x)
	    (setf (self repeats) (assure-tensor (!shape x (self axis))))
	    (with-searching-calc-node :sum x (self axis)))
  :backward ((dy)
	     (list (!div (!repeats dy (self axis) (self repeats))
			 (self repeats)))))

(defnode RepeatTensor (axis repeats) 
  :parameters ((axis axis) (repeats repeats))
  :forward ((x) (with-searching-calc-node :repeat x (self axis) (self repeats)))
  :backward ((dy) (list (!sum dy (self axis)))))

(defnode ExpTensor ()
  :parameters ((xi T))
  :forward ((x) (setf (self xi) x)
		(with-searching-calc-node :exp x))
  :backward ((dy)
	     (list (!mul dy (!exp (self xi))))))

(defnode MatMulTensor ()
  :parameters ((xi T) (yi T))
  :forward ((x y) (setf (self xi) x)
		  (setf (self yi) y)
		  (with-searching-calc-node :matmul x y))
  :backward ((dy)
	     (list (!matmul dy (!transpose (self yi)))
		   (!matmul (!transpose (self xi)) dy))))

(defun !add (x y)
  ;from now on, static typing
  (call (AddTensor) (assure-tensor x) (assure-tensor y)))
    
(defun !sub (x y)
  (call (SubTensor) (assure-tensor x) (assure-tensor y)))

(defun !mul (x y)
  (call (MulTensor) (assure-tensor x) (assure-tensor y)))

(defun !div-old (x y)
  (unless (= x 1) (error "!div-old: x must be 1"))
  ; x must be 1, cl-waffe.backends.mgl:div has some problems?...
  (call (DivTensor) (assure-tensor x) (assure-tensor y)))

(defun !div (x y)
  ; its faster
  (!mul x (!div-old 1 y)))
  
(defun !dot (x y)
  (call (DotProductTensor) (assure-tensor x) (assure-tensor y)))

(defun !sum (x &optional (axis nil) (keepdims nil))
  (if (null axis)
      (let ((axis-size (!dims x))
	    (result x))
	(dotimes (i axis-size)
	  (setq result (!sum result (1- (- axis-size i)))))
	result)
      (let ((nrepeat (!shape x axis))
	    (result (call (SumTensor (assure-tensor axis) (if (null keepdims) -1 1)) (assure-tensor x))))
	(if keepdims
	    (!repeats (!unsqueeze result axis) axis nrepeat)
	    result))))

(defun !mean (x &optional (axis nil) (keepdims nil))
  (let ((nrepeat (!shape x axis))
	(result (call (MeanTensor (assure-tensor axis)) (assure-tensor x))))
    (if keepdims
	(!repeats (!unsqueeze result axis) axis nrepeat)
	result)))

(defun !pow (x n)
  (call (PowTensor) (assure-tensor x) (assure-tensor n)))

(defun !sqrt (x)
  (call (SqrtTensor) (assure-tensor x)))

(defun !log (x)
  (call (LogTensor) (assure-tensor x)))

(defun !reshape (x dim)
  (call (ReshapeTensor (assure-tensor dim)) (assure-tensor x)))

(defun !repeats (x axis repeats)
  (call (RepeatTensor (assure-tensor axis) (assure-tensor repeats)) (assure-tensor x)))

(defun !transpose (x &optional result)
  (call (TransposeTensor (assure-tensor (if result (numcl:asarray result)))) (assure-tensor x)))

(defun !matmul (x y)
  (cond
    ((and (= (!dims x) (!dims y))
	  (= (!dims x) 2))
     (call (MatMulTensor) (assure-tensor x) (assure-tensor y)))
    ((and (= (!dims x) 1) (= (!dims y) 2))
     (call (MatMulTensor) (!unsqueeze (assure-tensor x) -1) (assure-tensor y)))
    ((and (= (!dims x) 2) (= (!dims y) 1))
     (call (MatMulTensor) (assure-tensor x) (!unsqueeze (assure-tensor y) -1)))
    (T (error "matmul for 3d/4d tensor is coming soon..."))))

(defun !unsqueeze (x &optional (dim 0))
  ; display error when (!dims x) >= dim
  (let ((s (!shape x)))
    (case dim
      (0  (setq s `(1 ,@s)))
      (-1 (push 1 (cdr (nthcdr (1- (length s)) s))))
      (T  (push 1 (cdr (nthcdr (1- dim) s)))))
    (!reshape x s)))

(defun !squeeze (x &optional (dim nil))
  (labels ((remove-nth (nth list)
	     (loop for i in list
		   for idx from 0
		   unless (= idx nth)
		     collect i)))
    (let ((s (!shape x)))
      (cond
	((null dim) (setq s (remove 1 s)))
	((eq dim 0) (setq s (if (= (car s) 1)
		       (cdr s)
		       s)))
	((eq dim -1) (setq s (if (= (car (last s)) 1)
			(butlast s)
			s)))
	(T (setq s (if (= (nth dim s) 1)
		       (remove-nth dim s)
		       s))))
      (!reshape x s))))

(defun !exp (x)
  (call (ExpTensor) (assure-tensor x)))


