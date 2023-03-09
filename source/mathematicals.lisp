
(in-package :cl-waffe)


(defnode PowTensor nil
  :optimize t
  :parameters ((xi T) (yi T))
  :forward ((x1 y1)
	    (save-for-backward xi x1)
	    (save-for-backward yi y1)
	    (with-searching-calc-node :pow x1 y1))
  :backward ((dy)
	     (list (!mul (!mul dy (self yi))
			 (!pow (self xi) (- (the single-float (data (self yi))) 1)))
		   (!mul (!mul
			  (!log (self xi))
			  (!pow (self xi) (self yi)))
			 dy))))

(defun !!pow (target-x n)
  "Takes the power of each element in @cl:param(x) with n.

target-x is destructed."

  (let ((target-x (assure-tensor target-x)))
    (!allow-destruct target-x)
    (!pow target-x n)))

(defope !pow (PowTensor) node (x n)
    "Takes the power of each element in @cl:param(x) with n, returning a new sysconst.

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!ones `(10 10)))
(!pow a 3)
;#Const(((1.0 1.0 ~ 1.0 1.0)        
;                 ...
;        (1.0 1.0 ~ 1.0 1.0)) :mgl t :shape (10 10))
@end[lang=lisp](code)
@end(section)"
  (call node (assure-tensor x) (assure-tensor n)))


(defnode SqrtTensor nil
  :optimize t
  :parameters ((xi T))
  :forward ((x1) (save-for-backward xi x1)
		 (with-searching-calc-node :sqrt x1))
  :backward ((dy)
	     (list (!div dy (!mul (!sqrt (self xi)) 2)))))

(defope !sqrt (SqrtTensor) node (x)
    "Takes the power of each element in @cl:param(x) with 1/2, creating new sysconst and nodes.

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!ones `(10 10)))
(!sqrt a 3)
;#Const(((1.0 1.0 ~ 1.0 1.0)
;                 ...
;        (1.0 1.0 ~ 1.0 1.0)) :mgl t :shape (10 10))
@end[lang=lisp](code)
@end(section)"
  (call node (assure-tensor x)))

(defun !!sqrt (target-x)
  "Takes the power of each element in @cl:param(x) with 1/2.

target-x is destructed."
  (let ((target-x (assure-tensor target-x)))
    (!allow-destruct target-x)
    (!sqrt target-x)))


(defnode LogTensor nil
  :optimize t
  :parameters ((x1 T))
  :forward ((x1) (save-for-backward x1 x1)
		 (with-searching-calc-node :log x1))
  :backward ((dy) (list (!div dy (self x1)))))

(defun !!log (target-x)
  "Returns a modified tenssor with the natural logarithm of the elements of target-x"
  (let ((target-x (assure-tensor target-x)))
    (!allow-destruct target-x)
    (!log target-x)))

(defope !log (LogTensor) node (x)
    "Returns a new tensor with the natural logarithm of the elements of input.

yi = log(e xi)

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!ones '(10 10)))
(!log a)
;#Const(((0.0 0.0 ~ 0.0 0.0)        
;                 ...
;        (0.0 0.0 ~ 0.0 0.0)) :mgl t :shape (10 10))
@end[lang=lisp](code)
@end(section)"
  (call node (assure-tensor x)))


(defnode ExpTensor ()
  :optimize t
  :parameters ((xi T))
  :forward ((x) (save-for-backward xi x)
		(with-searching-calc-node :exp x))
  :backward ((dy)
	     (list (!mul (!exp (self xi)) dy))))

(defun !!exp (target-x)
  "Applying !exp in a destructive way."
  (let ((target-x (assure-tensor target-x)))
    (!allow-destruct target-x)
    (call (ExpTensor) target-x)))

(defope !exp (ExpTensor) node (x)
    "Applying exp to each element of x, creating a new sysconst.

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!randn `(10 10)))
;#Const(((0.624... 0.807... ~ 0.500... 0.937...)        
;                 ...
;        (0.662... 0.299... ~ 0.761... 0.729...)) :mgl t :shape (10 10))
(!exp a)
;#Const(((1.866... 2.242... ~ 1.650... 2.553...)        
;                 ...
;        (1.939... 1.349... ~ 2.140... 2.073...)) :mgl t :shape (10 10))
@end[lang=lisp](code)
@end(section)"
  
  (call node (assure-tensor x)))


; Trigonometric functions.

(defnode SinTensor ()
  :optimize t
  :parameters ((xi nil))
  :forward ((x)
	    (save-for-backward xi x)
	    (with-searching-calc-node :sin x))
  :backward ((dy)
	     (list (!mul dy (!cos (self xi))))))

(defnode CosTensor ()
  :optimize t
  :parameters ((xi nil))
  :forward ((x)
	    (save-for-backward xi x)
	    (with-searching-calc-node :cos x))
  :backward ((dy)
	     (list (!mul dy (!mul -1.0 (!sin (self xi)))))))

(defnode TanTensor ()
  :optimize t
  :parameters ((xi nil))
  :forward ((x)
	    (save-for-backward xi x)
	    (with-searching-calc-node :tan x))
  :backward ((dy)
	     (list (!mul dy (!div 1 (!pow (!cos (self xi)) 2))))))

(defnode ASinTensor ()
  :parameters ((xi nil))
  :forward ((x)
	    (save-for-backward xi x)
	    (with-searching-calc-node :asin x))
  :backward ((dy)
	     (list (!mul dy (!acos (self xi))))))

(defnode ACosTensor ()
  :parameters ((xi nil))
  :forward ((x)
	    (save-for-backward xi x)
	    (with-searching-calc-node :acos x))
  :backward ((dy)
	     (list (!mul dy (!mul -1.0 (!asin (self xi)))))))

(defnode ATanTensor ()
  :parameters ((xi nil))
  :forward ((x)
	    (save-for-backward xi x)
	    (with-searching-calc-node :atan x))
  :backward ((dy)
	     (list (!mul dy (!div 1 (!pow (!acos (self xi)) 2))))))



(defnode ASinhTensor ()
  :parameters ((xi nil))
  :forward ((x)
	    (save-for-backward xi x)
	    (with-searching-calc-node :asinh x))
  :backward ((dy)
	     (list (!mul dy (!acosh (self xi))))))

(defnode ACoshTensor ()
  :parameters ((xi nil))
  :forward ((x)
	    (save-for-backward xi x)
	    (with-searching-calc-node :acosh x))
  :backward ((dy)
	     (list (!mul dy (!mul -1.0 (!asinh (self xi)))))))

(defnode ATanhTensor ()
  :parameters ((xi nil))
  :forward ((x)
	    (save-for-backward xi x)
	    (with-searching-calc-node :atanh x))
  :backward ((dy)
	     (list (!mul dy (!div 1 (!pow (!acosh (self xi)) 2))))))

(defnode HyperbolicSinTensor ()
  :optimize t
  :parameters ((xi nil))
  :forward ((x)
	    (save-for-backward xi x)
	    (with-searching-calc-node :sinh x))
  :backward ((dy)
	     (list (!mul dy (!cosh (self xi))))))

(defnode HyperbolicCosTensor ()
  :optimize t
  :parameters ((xi nil))
  :forward ((x)
	    (save-for-backward xi x)
	    (with-searching-calc-node :cosh x))
  :backward ((dy)
	     (list (!mul dy (!sinh (self xi))))))


(defope !sin (SinTensor) node (x)
    "Applying sin to each element of x, creating a new sysconst.

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!randn `(5)))
;=>#Const((0.638... 0.527... 0.515... 0.495... 0.912...) :mgl t :shape (5))
(!sin a)
;=>#Const((-0.44... -0.64... -0.66... -0.70... -0.09...) :mgl t :shape (5))
@end[lang=lisp](code)
@end(section)"

  (call node (assure-tensor x)))

(defope !cos (CosTensor) node (x)
    "Applying cos to each element of x, creating a new sysconst.

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!randn `(5)))
;=>#Const((0.638... 0.527... 0.515... 0.495... 0.912...) :mgl t :shape (5))
(!cos a)
;=>#Const((0.803... 0.864... 0.870... 0.879... 0.611...) :mgl t :shape (5))
@end[lang=lisp](code)
@end(section)"

  (call node (assure-tensor x)))

(defope !tan (TanTensor) node (x)
    "Applying tan to each element of x, creating a new sysconst.

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!randn `(5)))
;=>#Const((0.638... 0.527... 0.515... 0.495... 0.912...) :mgl t :shape (5))
(!tan a)
;=>#Const((0.741... 0.582... 0.566... 0.540... 1.293...) :mgl t :shape (5))
@end[lang=lisp](code)
@end(section)"

  (call node (assure-tensor x)))

(defope !sinh (HyperbolicSinTensor) node (x)
    "Applying sinh to each element of x, creating a new sysconst.

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!randn `(5)))
;=>#Const((0.638... 0.527... 0.515... 0.495... 0.912...) :mgl t :shape (5))
(!sinh a)
;=>#Const((0.682... 0.551... 0.538... 0.516... 1.044...) :mgl t :shape (5))
@end[lang=lisp](code)
@end(section)"

  (call node (assure-tensor x)))

(defope !cosh (HyperbolicCosTensor) node (x)
    "Applying cosh to each element of x, creating a new sysconst.

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!randn `(5)))
;=>#Const((0.638... 0.527... 0.515... 0.495... 0.912...) :mgl t :shape (5))
(!cosh a)
;=>#Const((1.210... 1.142... 1.135... 1.125... 1.446...) :mgl t :shape (5))
@end[lang=lisp](code)
@end(section)"

  (call node (assure-tensor x)))

(defun !asin (x)
  "Applying asin to each element"
  (call (ASinTensor) (assure-tensor x)))

(defun !acos (x)
  "Applying acos to each element"
  (call (ACosTensor) (assure-tensor x)))

(defun !atan (x)
  "Applying atan to each element"
  (call (ATanTensor) (assure-tensor x)))

(defun !asinh (x)
  "Applying asinh to each element"
  (call (ASinhTensor) (assure-tensor x)))

(defun !acosh (x)
  "Applying acosh to each element"
  (call (ACoshTensor) (assure-tensor x)))

(defun !atanh (x)
  "Applying atanh to each element"
  (call (ATanhTensor) (assure-tensor x)))


(defnode AbsTensor ()
  :optimize t
  :parameters ((mask nil))
  :forward ((x)
	    (let ((mask (!where #'(lambda (x)
				    (declare (type single-float x))
				    (> x 0.0))
				x 1.0 -1.0)))
	      (save-for-backward mask x)
	      (!mul x mask)))
  :backward ((dy)
	     (list (!mul dy (self mask)))))

(defope !abs (AbsTensor) node (x)
    "Computes the absolute value of each element in @cl:param(x).

Example:
@begin[lang=lisp](code)
(setq a (!random `(10 10) '(-1.0 1.0)))
;#Const(((0.048... 0.805... ~ 0.769... 0.252...)        
;                 ...
;        (0.159... -0.66... ~ -0.55... -0.23...)) :mgl t :shape (10 10))
(!abs a)
;#Const(((0.048... 0.805... ~ 0.769... 0.252...)        
;                 ...
;        (0.159... 0.667... ~ 0.553... 0.239...)) :mgl t :shape (10 10))
@end[lang=lisp](code)"
  (call node (assure-tensor x)))

