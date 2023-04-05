
(in-package :cl-waffe)


(defparameter pi-single-float (the single-float (coerce pi 'single-float)))

(defnode ReLUTensor nil
  :parameters ((path-through nil)
	       (zero-buff nil))
  :forward-declaim (declaim (ftype (function (ReLUTensor waffetensor) waffetensor) :forward))
  :forward ((x)
	    (declare (optimize (speed 3)))
	    (unless (self zero-buff)
	      (setf (self zero-buff) (!zeros (!shape x))))
	    (let ((mask (with-searching-calc-node :< x (self zero-buff))))
	      (save-for-backward path-through mask)
	      (!mul mask x)))
  :backward-declaim (declaim (ftype (function (ReLUTensor waffetensor) list) :backward))
  :backward ((dy)
	     (list (!mul (self path-through) dy))))

(defun !relu (x)
  "Applying relu to x, return a new sysconst with making nodes.

Relu(x) = { 0 (x < 0), x (x > 0) }

Input: x where x is waffe supported data type.

Output: Tensor"
  (declare (optimize (speed 3))
	   (type waffetensor x))
  (call (ReLUTensor) (assure-tensor x)))

(defnode SigmoidTensor nil
  :parameters ((xi T))
  :forward ((x)
	    (let ((result (!!div 1 (!!add (!!exp (!mul -1 x)) 1))))
	      (save-for-backward xi result)
	      result))
  :backward ((dy) (list (!mul dy
			      (!mul (self xi)
				    (!sub 1 (self xi)))))))

(defun !sigmoid (x)
  "Applyong sigmoid to x, return a new sysconst with making nodes.

Input: x where x is waffe supported data type.

Output: Tensor"
  (call (SigmoidTensor) (assure-tensor x)))

(defnode TanhTensor nil
  :parameters ((xi T))
  :forward ((x)
	    (save-for-backward xi x)
	    (with-searching-calc-node :tanh x))
  :backward ((dy)
	     (list (!mul dy (!div 1 (!pow (!cosh (self xi)) 2))))))

(defun !tanh (x)
  "Applying tanh to x, return a new sysconst with making nodes."
  (call (TanhTensor) (assure-tensor x)))

; Optimizing won't go well
(define-with-typevar !gelu u (x &key (approximate t))
  "Applying gelu to x, returning a new sysconst.

Paper: https://arxiv.org/abs/1606.08415.

TOOD: Improve its performance

GeLU(x) = x * s(x)

When approximate is t:

s(x) = x/2 * [1 + tanh(sqrt(2/pi * (x + 0.044715 * x^3)))]

When is nil:

Not implemented (TODO)

@begin[lang=lisp](code)
(setq x (!randn `(10 10)))
(!gelu x)
;#Const(((0.201... 0.038... ~ 0.158... 0.040...)        
;                 ...
;        (0.300... 1.395... ~ 0.030... 0.029...)) :mgl t :shape (10 10))
@end[lang=lisp](code)"
  (declare (optimize (speed 3)))
  ; s(x) is not necessary derivable ↓ is is-ancestor-tensor considered?
  (let ((s (if approximate
	       (!mul (!div x 2)
		     (!filter x
			      #'(lambda (el)
				  (declare (type u el))
				  (multiple-value-bind (n)
				      ; failed to optimize
				      (floor
				       (the u
					    (* (sqrt (the (u 0e0)
							  (/ 2.0
							     (the u pi-single-float))))
					       (+ el
						  (* 0.044715
						     (expt el 3))))))
				    (the u (+ 1.0 (tanh n)))))))
	       (error "I'm sorry but its not implemented yet..."))))
    (!mul x s)))



(define-with-typevar !leakey-relu u (x &optional (alpha 0.01))
  "Applying Leakey-relu to x, returning a new sysconst.

Leakey-ReLU is defined as out = {alpha (x < 0), x (x >= 0)}

Example:

@begin[lang=lisp](code)
(setq x (!randn `(10 10)))
#Const(((0.635... -0.56... ~ -1.15... -1.50...)        
                 ...
        (0.775... 1.258... ~ -1.29... 0.240...)) :mgl t :shape (10 10))

(!leakey-relu x)
#Const(((0.635... 0.003... ~ 0.013... 0.022...)        
                 ...
        (0.775... 1.258... ~ 0.016... 0.240...)) :mgl t :shape (10 10))
@end[lang=lisp](code)"
  (declare (optimize (speed 3))
	   (type u alpha))
  (!mul x (!filter x #'(lambda (x)
			 (declare (type u x))
			 (if (>= x 0)
			     1.0
			     (* alpha x))))))

(defmodel Swish (&key (beta 1.0)
		      (trainable t))
  :parameters ((beta (if trainable
			 (tensor beta)
			 (const beta))))
  :forward ((x)
	    (!swish x :beta (self beta))))



(defun !swish (x &key (beta (const 1.0)))
  "Applying swish to each element of x

Swish is defined as out = (/ 1 (+ 1 (exp (* beta -1 x))))

In default beta is 1.0, if you want to use trainable one, @cl:param(Swish) is available as a waffe model.

Note that beta must begin given as a waffetensor.

@begin[lang=lisp](code)
(setq x (!randn `(10 10)))
#Const(((0.635... -0.56... ~ -1.15... -1.50...)        
                 ...
        (0.775... 1.258... ~ -1.29... 0.240...)) :mgl t :shape (10 10))

(!swish x)
;#Const(((0.415... -0.20... ~ -0.27... -0.27...)        
;                 ...
;        (0.531... 0.980... ~ -0.27... 0.134...)) :mgl t :shape (10 10))

(call (Swish :beta 1.0) x) ; its beta is trainable by backpropgating.
;#Const(((0.415... -0.20... ~ -0.27... -0.27...)        
;                 ...
;        (0.531... 0.980... ~ -0.27... 0.134...)) :mgl t :shape (10 10))
@end[lang=lisp](code)"
  (!div x (!add 1.0 (!exp (!mul (!mul -1.0 beta) x)))))

(defun !mish () "Todo")

