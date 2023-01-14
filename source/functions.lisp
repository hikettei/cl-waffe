
(in-package :cl-waffe)


(defnode ReLUTensor nil
  :optimize t
  :parameters ((path-through T) (zero-buff T))
  :forward ((x)
	    (if (equal (self zero-buff) T)
		(setf (self zero-buff) (!zeros (!shape x))))
	    (setf (self path-through) (with-searching-calc-node :< x (self zero-buff)))
	    (!mul (self path-through) x))
  :backward ((dy)
	     (list (!mul (self path-through) dy))))

(defun !relu (x)
  (call (ReLUTensor) (assure-tensor x)))

(defnode SigmoidTensor nil
  :optimize t
  :parameters ((xi T))
  :forward ((x)
	    (setf (self xi) x)
            (!div (!add 1 (!tanh (!div x 2))) (const 2)))
  :backward ((dy) (let ((p (!sigmoid (self xi))))
		    (list (!mul p (!mul dy (!sub 1 p)))))))

(defun !sigmoid (x)
  (call (SigmoidTensor) (assure-tensor x)))

(defnode TanhTensor nil
  :optimize t
  :parameters ((xi T))
  :forward ((x)
	    (setf (self xi) x)
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
  (let* ((x1 (if avoid-overflow
		(!sub x (!average x))
		x))
	 (z (!sum (!exp x1) 1 t)))
    (!div (!exp x1) z)))



