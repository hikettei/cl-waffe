
(in-package :cl-waffe)


(defnode TransposeTensor (shape)
  :parameters ((prev-shape T) (shape shape))
  :forward ((x)
	    (setf (self prev-shape) (assure-tensor (!shape x)))
	    (with-searching-calc-node :transpose x (self shape)))
  :backward ((d1)
	     (list (!transpose d1))))

(defnode TransposeOriginalTensor (shape)
  :parameters ((prev-shape nil) (shape shape))
  :forward ((x)
	    (setf (self prev-shape) (!shape x))
	    (with-facet (array ((value x) 'array :direction :input))
	      (sysconst (array-to-mat (numcl:transpose array)))))
  :backward ((dy)
	     (list (!transpose1 dy (self prev-shape)))))

(defun !transpose (x &optional result)
  "Transpose x where x is a 2d tensor.

Transposed x is lazy evaluated until called by !matmul.

Todo: implement 3d, 4d version...

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!randn `(3 5)))
(setq a (!transpose a))
;#Const(#<FUNCTION (LABELS CL-WAFFE.BACKENDS.MGL::LAZYTRANSPOSE :IN CL-WAFFE.BACKENDS.MGL::LAZY-EVAL-TRANSPOSE) {10038CBADB}>)

(!matmul a (!randn '(3 5)))
;#Const(((0.653... 0.400... 0.471... 0.705... 0.623...)        
;                 ...
;        (1.220... 0.760... 0.975... 1.360... 1.029...)) :mgl t :shape (5 5))
@end[lang=lisp](code)
@end(section)"
  (call (TransposeTensor (assure-tensor result)) (assure-tensor x)))

(defun !transpose1 (x &rest result)
  "Transpose x but doesn't produce lazy-eval.

Todo: Numcl's operation couldm't optimized well. i need to reimplement it by myself.

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!randn `(10 5 3)))

(!transpose1 a)
;#Const((((-0.47... -0.03... ~ -0.17... 0.328...)         
;                   ...
;         (0.210... -1.80... ~ 1.648... 0.135...))        
;                 ...
;        ((-0.52... 1.509... ~ 0.643... 0.258...)         
;                   ...
;         (-0.26... -1.14... ~ -1.08... 1.126...))) :mgl t :shape (3 5 10))
@end[lang=lisp](code)
@end(section)"
  (call (TransposeOriginalTensor (assure-tensor result)) (assure-tensor x)))

