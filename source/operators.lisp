
(in-package :cl-waffe)

(defun assure-tensor (x)
  (if (typep x 'WaffeTensor)
      x
      (const x)))

; 連続してcallopすると計算木が生成されない
(defmodel AddTensor nil
  :parameters nil
  :forward  ((x y) (callop :add x y))
  :backward ((dy) (values dy dy)))

(defmodel MulTensor nil
  :parameters nil
  :forward ((x y) (callop :mul x y))
  :backward ((dy) dy))

(defmethod add (x y)
  (call (AddTensor) (assure-tensor x) (assure-tensor y)))

(defmethod mul (x y)
  (call (MulTensor) (assure-tensor x) (assure-tensor y)))
