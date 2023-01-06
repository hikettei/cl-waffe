
(in-package :cl-waffe)


(defnode ReLUTensor nil
  :parameters ((path-through T) (zero-buff T))
  :forward ((x)
	    (if (equal (self zero-buff) T)
		(setf (self zero-buff) (!zeros (!shape x))))
	    (setf (self path-through) (callop :< x (self zero-buff)))
	    (callop :mul (self path-through) x))
  :backward ((dy)
	     (list (callop :mul (self path-through) dy))))

(defun !relu (x)
  (call (ReLUTensor) (assure-tensor x)))

(defnode SigmoidTensor nil
  :parameters ((xi T))
  :forward ((x)
	    (setf (self xi) x)
            (detach (!div (!add 1 (!tanh (!div x 2))) (const 2))))
  :backward ((dy) (let ((p (!sigmoid (self xi))))
		    (list (callop :mul p (!mul dy (!sub 1 p)))))))

(defun !sigmoid (x)
  (call (SigmoidTensor) (assure-tensor x)))

(defnode TanhTensor nil
  :parameters ((xi T))
  :forward ((x)
	    (setf (self xi) x)
	    (callop :tanh x))
  :backward ((dy)
	     (list (callop :mul dy (callop :sub (const 1) (!pow (callop :tanh (self xi)) 2))))))

(defun !tanh (x)
  (call (TanhTensor) (assure-tensor x)))

(defun !softmax (x) ; backward
  (let ((z (!sum (!exp x) 1 t)))
    (!div (!exp x) z)))


