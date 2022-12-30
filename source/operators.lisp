
(in-package :cl-waffe)

(defun assure-tensor (x)
  (if (typep x 'WaffeTensor)
      x
      (const x)))

; callopは計算ノードから切り離されてることに注意

(defnode AddTensor nil
  :parameters nil
  :forward  ((x y) (callop :add x y))
  :backward ((dy) (list dy dy)))

(defnode SubTensor nil
  :parameters nil
  :forward ((x y) (callop :sub x y))
  :backward ((dy) (list dy (mul dy -1))))

(defnode MulTensor nil
  :parameters ((xi T) (yi T))
  :forward ((x y)
	    (setf (self xi) x)
	    (setf (self yi) y)
	    (callop :mul x y))
  :backward ((dy) (list (callop :mul dy (self yi))
			(callop :mul dy (self xi)))))

(defnode DivTensor nil
  :parameters ((xi T) (yi T))
  :forward ((x y)
            (setf (self xi) x)
	    (setf (self yi) y)
	    (callop :div x y))
  :backward ((dy) (list (callop :div dy (self yi))
			(callop :div (mul (mul dy (self xi)) -1) (mul (self yi) (self yi))))))

(defnode PowTensor nil
  :parameters ((xi T) (yi T))
  :forward ((x1 y1) (setf (self xi) x1) (setf (self yi) y1)
		    (callop :pow x1 y1))
  :backward ((dy) (list (mul (mul dy (self x2)) (pow (self x1) (sub (self y1) 1)))
			(pow (mul dy (self y1)) (mul (self yi) (loge (self xi)))))))

(defnode LogTensor nil
  :parameters ((x1 T))
  :forward ((x1) (setf (self x1) x1) (callop :log x1))
  :backward ((dy) (list (callop :div dy (self x1)))))

(defnode ReshapeTensor (shape)
  :parameters ((prev-shape T) (shape shape))
  :forward ((x) (setf (self prev-shape) (shape x)) (callop :reshape x (self shape)))
  :backward ((dy) (list (reshape dy (self prev-shape)))))

(defnode DotProductTensor nil
  :parameters ((xi T) (yi T))
  :forward ((x1 x2) (setf (self xi) x1)
		    (setf (self yi) x2)
		    (callop :dot x1 x2))
  :backward ((dy)
	     (list (dot dy (transpose (self yi))) (dot (transpose (self yi)) dy))))

(defnode TransposeTensor (shape)
  :parameters ((prev-shape T) (shape shape))
  :forward ((x) (setf (self prev-shape) (shape x)) (callop :transpose x (self shape)))
  :backward ((d1) (callop :transpose d1 (self prev-shape))))

(defnode MeanTensor (axis)
  :parameters ((axis axis) (repeats T))
  :forward ((x)
	    (setf (self repeats) (assure-tensor (shape x axis)))
	    (callop :mean x (self axis)))
  :backward ((dy) (list (repeats dy (self axis) (self repeats)))))

(defnode SumTensor (axis)
  :parameters ((axis axis) (repeats T))
  :forward ((x)
	    (setf (self repeats) (assure-tensor (shape x axis)))
	    (callop :sum x (self axis)))
  :backward ((dy) (list (callop :div
				 (repeats dy (self axis) (self repeats))
				 (self repeats)))))

(defnode RepeatTensor (axis repeats)
  :parameters ((axis axis) (repeats repeats))
  :forward ((x) (callop :repeat x (self axis) (self repeats)))
  :backward ((dy) (list (sum dy (self axis)))))

(defnode ExpTensor nil
  :parameters ((xi T))
  :forward ((x) (setf (self xi) x) (callop :exp x))
  :backward ((dy) (list (mul dy (t-exp (self xi))))))

;ScalarMul

(defun add (x y)
  (call (AddTensor) (assure-tensor x) (assure-tensor y)))
    
(defun sub (x y)
  (call (SubTensor) (assure-tensor x) (assure-tensor y)))

(defun mul (x y)
  (call (MulTensor) (assure-tensor x) (assure-tensor y)))

(defun div (x y)
  (call (DivTensor) (assure-tensor x) (assure-tensor y)))

(defun dot (x y)
  (call (DotProductTensor) (assure-tensor x) (assure-tensor y)))

(defun sum (x &optional (axis nil))
  (call (SumTensor (assure-tensor axis)) (assure-tensor x)))

(defun pow (x n)
  (call (PowTensor) (assure-tensor x) (assure-tensor n)))

(defun mean (x &optional (axis nil))
  (call (MeanTensor (assure-tensor axis)) (assure-tensor x)))

(defun loge (x)
  (call (LogTensor) (assure-tensor x)))

(defun reshape (x dim)
  (call (ReshapeTensor (assure-tensor dim)) (assure-tensor x)))

(defun repeats (x axis repeats)
  (call (RepeatTensor (assure-tensor axis) (assure-tensor repeats)) (assure-tensor x)))

(defun transpose (x &optional result)
  (call (TransposeTensor (assure-tensor result)) (assure-tensor x)))

(defun matmul (x y)
  ; 4 3d tensor
  (call (DotProductTensor) (assure-tensor x) (assure-tensor y)))

(defun t-exp (x)
  (call (ExpTensor) (assure-tensor x)))


