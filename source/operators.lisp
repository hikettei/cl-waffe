
(in-package :cl-waffe)

(defun assure-tensor (x)
  (if (typep x 'WaffeTensor)
      x
      (const x)))

; 連続してcallopすると計算木が生成されない

(defmodel AddTensor nil
  :parameters nil
  :forward  ((x y) (callop :add x y))
  :backward ((d1 d2) (list d1 d2)))

(defmodel MulTensor nil
  :parameters ((xi T) (yi T))
  :forward ((x y)
	    (setf (self xi) x)
	    (setf (self yi) y)
	    (callop :mul x y))
  :backward ((d1 d2) (list (callop :mul d1 (self yi))
			   (callop :mul d2 (self xi)))))

(defun add (x y)
  (call (AddTensor) (assure-tensor x) (assure-tensor y)))

(defun mul (x y)
  (call (MulTensor) (assure-tensor x) (assure-tensor y)))
