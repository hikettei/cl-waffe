
(in-package :cl-waffe.nn)


(defmodel LinearLayer (in-features out-features &optional (bias T))
  :document "Calling LinearLayer.
Applies a linear transformation to the coming datum. y = xA + b

Args:  in-features (fixnum)
       out-features (fixnum)
       bias (boolean) (See LinearLayer's document)

Input: x (Tensor) where the x is the shape of (batch-size in-features)
Output: Applied tensor, where the tensor is the shape of (batch-size out-features)"
  :optimize t
  :parameters ((weight
		(parameter (!mul 0.01 (!randn `(,in-features ,out-features))))
		:type waffetensor)
	      (bias (if bias
			(parameter (!zeros `(1 ,out-features)))
			nil)))
  :forward ((x)
	    (cl-waffe.nn:linear x (self weight) (self bias))))

(defmodel DenseLayer (in-features out-features &optional (bias T) (activation :relu))
  :document "Calling LinearLayer, and activation.
Args:  in-features (fixnum)
       out-features (fixnum)
       bias (boolean) (See LinearLayer's document)

       activation: (symbol or function)
           the symbol is following: :relu :sigmoid :tanh :softmax
           when the activation is function, call this as activation.
Input: x (Tensor) where the x is the shape of (batch-size in-features)
Output: Applied tensor, where the tensor is the shape of (batch-size out-features)
"
  :optimize t
  :parameters ((layer (linearlayer in-features out-features bias)) (activation activation))
  :forward ((x)
	    (case (cl-waffe:self activation)
	      (:relu
	       (!relu (call (cl-waffe:self layer) x)))
	      (:sigmoid
	       (!sigmoid (call (cl-waffe:self layer) x)))
	      (:tanh
	       (!tanh (call (cl-waffe:self layer) x)))
	      (:softmax
	       (!softmax (call (cl-waffe:self layer) x)))
	      (T
	       (funcall (the function (cl-waffe:self activation))
			(call (cl-waffe:self layer) x))))))
