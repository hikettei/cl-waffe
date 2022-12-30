
(in-package :cl-waffe)

(defun assure-tensor (x)
  (if (typep x 'WaffeTensor)
      x
      (const x)))

; callopは計算ノードから切り離されてることに注意

(defmodel AddTensor nil
  :parameters nil
  :forward  ((x y) (callop :add x y))
  :backward ((d1 d2) (list d1 d2)))

(defmodel SubTensor nil
  :parameters nil
  :forward ((x y) (callop :sub x y))
  :backward ((d1 d2) (list d1 (mul d2 -1))))

(defmodel MulTensor nil
  :parameters ((xi T) (yi T))
  :forward ((x y)
	    (setf (self xi) x)
	    (setf (self yi) y)
	    (callop :mul x y))
  :backward ((d1 d2) (list (callop :mul d1 (self yi))
			   (callop :mul d2 (self xi)))))

(defmodel DivTensor nil
  :parameters ((xi T) (yi T))
  :forward ((x y)
            (setf (self xi) x)
	    (setf (self yi) y)
	    (callop :div x y))
  :backward ((d1 d2) (list (callop :div d1 (self yi))
			   (callop :div (mul (mul d2 (self xi)) -1) (mul (self yi) (self yi))))))

(defmodel PowTensor nil
  :parameters ((xi T) (yi T))
  :forward ((x1 y1) (setf (self xi) x1) (setf (self yi) y1)
		    (callop :pow x1 y1))
  :backward ((d1 d2) (list (mul (mul d1 (self x2)) (pow (self x1) (sub (self y1) 1)))
			   (pow (mul d2 (self y1)) (mul (self yi) (loge (self xi)))))))

(defmodel LogTensor nil
  :parameters ((x1 T))
  :forward ((x1) (setf (self x1) x1) (callop :log x1))
  :backward ((dy) (list (callop :div dy (self x1)))))

(defmodel ReshapeTensor (shape)
  :parameters ((prev-shape T) (shape shape))
  :forward ((x) (setf (self prev-shape) (shape x)) (callop :reshape x (self shape)))
  :backward ((dy) (list (reshape dy (self prev-shape)))))

(defmodel DotProductTensor nil
  :parameters ((xi T) (yi T))
  :forward ((x1 x2) (setf (self xi) x1)
		    (setf (self yi) x2)
		    (callop :dot x1 x2))
  :backward ((d1 d2)
	     (list (dot d1 (transpose (self yi))) (dot d2 (transpose (self xi))))))

(defmodel TransposeTensor (shape)
  :parameters ((prev-shape T) (shape shape))
  :forward ((x) (setf (self prev-shape) (shape x)) (callop :transpose x (self shape)))
  :backward ((d1) (callop :transpose d1 (self prev-shape))))

(defmodel MeanTensor (axis)
  :parameters ((axis axis) (repeats T))
  :forward ((x)
	    (setf (self repeats) (assure-tensor (shape x axis)))
	    (callop :mean x (self axis)))
  :backward ((dy) (list (repeats dy (self axis) (self repeats)))))

(defmodel SumTensor (axis)
  :parameters ((axis axis) (repeats T))
  :forward ((x)
	    (setf (self repeats) (assure-tensor (shape x axis)))
	    (callop :sum x (self axis)))
  :backward ((dy) (list (callop :div
				 (repeats dy (self axis) (self repeats))
				 (self repeats)))))

(defmodel RepeatTensor (axis repeats)
  :parameters ((axis axis) (repeats repeats))
  :forward ((x) (callop :repeat x (self axis) (self repeats)))
  :backward ((dy) (list (sum dy (self axis)))))

(defmodel ExpTensor nil
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
