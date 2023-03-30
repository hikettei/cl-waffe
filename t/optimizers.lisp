
(in-package :cl-waffe-test)

(in-suite :test)
#|
Here's tests for several activation, several optimizers, some operators.
Using dummy data, i will get assure activation and optimizers are working well.
|#

(defparameter size 200)
; preparing dummy data
(defparameter train (!beta `(,size ,(* 28 28)) 2.0 2.0))
(defparameter label (!zeros `(,size 10)))

(dotimes (i size)
  (setf (!aref label i (random 3)) (!ones `(1 1))))
; proablaly acc -> 1/3		 

(defmodel MLP (activation)
  :parameters ((layer1   (denselayer (* 28 28) 512 t activation))
	       (layer2   (denselayer 512 256 t activation))
	       (layer3   (linearlayer 256 10 t)))
  :forward ((x)
	    (with-calling-layers x
	      (layer1 x)
 	      (layer2 x)
	      (layer3 x))))

(print-model (MLP :tanh))

(defmacro define-test-trainer (name optim lr activation)
  `(progn
   (deftrainer ,name nil
     :model (MLP ,activation)
     :optimizer ,optim
     :optimizer-args (:lr ,lr)
     :step-model ((x y)
		  (zero-grad)
		  (let ((out (cl-waffe.nn:softmax-cross-entropy (call (model) x) y)))
		    (backward out)
		    (update)
		    out))
     :predict ((x) (call (model) x)))

   (defmethod print-object ((model ,name) stream)
     (format stream "[Trainer: ~a]" ',name))))

(defun test-for (trainer &optional (niter 3))
  (let ((losses nil))
    (print (slot-value trainer 'cl-waffe::optimizer))
    ; iterate for epoch
    (dotimes (epoch niter)
      (!reset-batch train)
      (!reset-batch label)
      (format t "~%Training ~a th epoch at ~a~%" epoch trainer)

      ; iterate for batch. todo -> !Loop-for-batch macro
      (loop for index fixnum upfrom 0 below (!shape train 0) by 100
	    do (progn
		 (!set-batch train index 100)
		 (!set-batch label index 100)
		 (push (step-model trainer train label) losses))))

    (< (data (car losses)) (data (car (last losses))))))

(define-test-trainer Trainer0 cl-waffe.optimizers:SGD 1e-2 :relu)
(define-test-trainer Trainer1 cl-waffe.optimizers:Momentum 1e-2 :relu)
(define-test-trainer Trainer2 cl-waffe.optimizers:AdaGrad 1e-4 :relu)
(define-test-trainer Trainer3 cl-waffe.optimizers:RMSProp 1e-4 :relu)
(define-test-trainer Trainer4 cl-waffe.optimizers:Adam 1e-3 :relu)
(define-test-trainer Trainer5 cl-waffe.optimizers:Adam 1e-3 :tanh)
(define-test-trainer Trainer6 cl-waffe.optimizers:Adam 1e-3 :sigmoid)
(define-test-trainer Trainer7 cl-waffe.optimizers:Adam 1e-3 #'!leakey-relu)
(define-test-trainer Trainer8 cl-waffe.optimizers:Adam 1e-4 #'!gelu)
(define-test-trainer Trainer9 cl-waffe.optimizers:Adam 1e-3 #'!swish)

(with-dtype :float
  (test test-training-in-float
	(is (test-for (Trainer0)))
	(is (test-for (Trainer1)))
	(is (test-for (Trainer2)))
	(is (test-for (Trainer3)))
	(is (test-for (Trainer4)))
	(is (test-for (Trainer5)))
	(is (test-for (Trainer6)))
	(is (test-for (Trainer7)))
	(is (test-for (Trainer8)))
	(is (test-for (Trainer9)))))

(with-dtype :double
  (test test-training-in-double
	(is (test-for (Trainer0)))
	(is (test-for (Trainer1)))
	(is (test-for (Trainer2)))
	(is (test-for (Trainer3)))
	(is (test-for (Trainer4)))
	(is (test-for (Trainer5)))
	(is (test-for (Trainer6)))
	(is (test-for (Trainer7)))
	(is (test-for (Trainer8)))
	(is (test-for (Trainer9)))))
