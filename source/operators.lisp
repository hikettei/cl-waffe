
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

;ScalarMul

(defun add (x y)
  (call (AddTensor) (assure-tensor x) (assure-tensor y)))

(defun mul (x y)
  (call (MulTensor) (assure-tensor x) (assure-tensor y)))

(defun sum (x &optional (axis nil))
  (call (SumTensor (assure-tensor axis)) (assure-tensor x)))

(defun repeats (x axis repeats)
  (call (RepeatTensor (assure-tensor axis) (assure-tensor repeats)) (assure-tensor x)))

(defun matmul (x y)
  (call (MatMulTensor) (assure-tensor x) (assure-tensor y)))

