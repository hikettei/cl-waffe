
(in-package :cl-waffe)

(defnode DotProductTensor nil
  :optimize t
  :parameters ((xi T) (yi T))
  :forward ((x1 x2) ; only supports 2d and 2d arrays
		    (save-for-backward xi x1)
		    (save-for-backward yi x2)
		    (with-searching-calc-node :dot x1 x2))
  :backward ((dy)
	     (list (!dot dy (!transpose (self yi)))
		   (!dot (!transpose (self xi)) dy))))

(defope !dot (DotProductTensor) node (x y)
    "Computes the dot product of x and y where x and y are 1d Tensor.

🗒Note: Unlike Numpy's dot, !dot only supports for 1d tensors with the same number of elements and the tensor of which dims is larger than 1, regarded as 1d tensors.

@begin(section)
@title(Example)
@begin[lang=lisp](code)

(setq a (!randn `(10)))
(setq b (!randn `(10)))

(!dot a b)
;=> #Const(1.0842022e-19)
@end[lang=lisp](code)
@end(section)
"
  (call node (assure-tensor x) (assure-tensor y)))

(defnode MatMulTensor ()
  :optimize t
  :parameters ((xi nil) (yi nil))
  :forward ((x y) (save-for-backward xi x)
		  (save-for-backward yi y)
		  (with-searching-calc-node :matmul x y))
  :backward ((dy)
	     (list (!matmul dy (!transpose (self yi)))
		   (!matmul (!transpose (self xi)) dy))))

(defope !matmul (MatmulTensor) node (x y)
    "Multiplying matrices @cl:param(x) and @cl:param(y).

!matmul has many behaviours depends on the dimensionality of the tensors as follows:

@begin(deflist)
@def(x and y are 1D)
@begin(term)
The dot-product is returned.
@begin[lang=lisp](code)
(setq a (!randn `(10)))
(setq b (!randn `(10)))
(!matmul a b)
;=>#Const(-2.0)
@end[lang=lisp](code)
@end(term)

@def(x and y are both 2D)
@begin(term)
The matrix-matrix product is returned.
@begin[lang=lisp](code)
(setq a (!randn `(3 10)))
(setq b (!randn `(10 3)))
(!matmul a b)
;#Const(((2.309... 2.223... 3.630...)        
;                 ...
;        (2.334... 2.850... 3.678...)) :mgl t :shape (3 3))
@end[lang=lisp](code)
@end(term)

@def(x is 2D and y is 3D.)
@begin(term)
The matrix and y's each matrix are multiplied and is returned.
@begin[lang=lisp](code)
(setq a (!randn `(3 10)))
(setq b (!randn `(5 10 3)))

(!matmul a b)
;(!aref b 0) ~ (!aref b 4) is multiplied with a

;#Const((((3.257... 2.731... 1.670...)         
;                   ...
;         (2.523... 2.251... 1.276...))        
;                 ...
;        ((2.610... 2.764... 2.415...)         
;                   ...
;         (2.080... 2.204... 1.751...))) :mgl t :shape (5 3 3))
@end[lang=lisp](code)
@end(term)

@def(x is 3D and y is 2D.)
@begin(term)
The matrix and x's each matrix are multiplied and is returned.
@begin[lang=lisp](code)
(setq a (!randn `(5 3 10)))
(setq b (!randn `(10 3)))

(!matmul a b)
;(!aref a 0) ~ (!aref a 4) is multiplied with b
;#Const((((2.309... 2.204... 1.556...)         
;                   ...
;         (3.746... 3.869... 3.091...))        
;                 ...
;        ((3.260... 3.200... 2.847...)         
;                   ...
;         (3.008... 2.186... 2.376...))) :mgl t :shape (5 3 3))
@end[lang=lisp](code)
@end(term)

@def(x is 3D and y is 3D.)
@begin(term)
The Batch Filtered Matrix-Matrix product is returned.
@begin[lang=lisp](code)
(setq a (!randn `(5 3 10)))
(setq b (!randn `(5 10 3)))

; The returned mat is comprised of:
; (!matmul (!aref a 0) (!aref b 0))
; (!matmul (!aref a 1) (!aref b 1))
; (!matmul (!aref a 2) (!aref b 2))
; (!matmul (!aref a 3) (!aref b 3))

(!matmul a b)
;#Const((((6.621... -5.61... 2.898...)         
;                   ...
;         (-2.96... -4.26... -3.99...))        
;                 ...
;        ((-0.02... 2.707... 5.989...)         
;                   ...
;         (-3.35... 3.561... -3.90...))) :mgl t :shape (5 3 3))
@end[lang=lisp](code)
@end(term)

@def(Otherwise)
@term(Currently not implemented. In the near future for more will be added.)
@end(deflist)"
  (cond
    ((and (= (the fixnum (!dims x)) 1)
	  (= (the fixnum (!dims y)) 1))
     (!dot x y))
    (T (call node (assure-tensor x) (assure-tensor y)))))



(defun !argmaxmin (tensor max-or-min &key (dim nil))
  "Todo: For GPU"
  (declare (optimize (speed 3))
	   (type waffetensor tensor))
  (let* ((dim (if (and dim (< (the fixnum dim) 0))
		  (the fixnum (+ (the fixnum (!dims tensor))
				 (the fixnum dim)))
		  dim))
	 (result (!zeros (if (null dim)
			     '(1)
			     (or (loop for i fixnum upfrom 0 below (!dims tensor)
				       while (< i (the fixnum dim))
				       collect (!shape tensor i))
				 `(1)))))
	 (iter-num (/ (the fixnum (!size tensor))
		      (the fixnum (!size result))))
	 (dim (or dim 0)))
    (if (>= (the fixnum (or dim 0))
	    (the fixnum (!dims tensor)))
	(error "!argmax/min: the specified dim ~a is larger than ~a. Satisfy :dim (~a) < !dims (~a)"
	       dim
	       (!dims tensor)
	       dim
	       (!dims tensor)))
    (with-facet (return-array ((data result) 'array :direction :output))
      (labels ((apply-tensor (rest-dims apply-dims)
		 (loop for i fixnum upfrom 0 below (car rest-dims)
		       do (if (= (length rest-dims) 1)
					; the tensor now referring is the last.
			      (let* ((m-val nil)
				     (m-pos nil)
				     (ts (loop for i fixnum upfrom 0 below dim
					       collect t))
				     (result-dim `(,@apply-dims ,i))
				     (result-dim1 `(,@ts ,@(cdr result-dim)))
				     (result-dim1 (if (= dim 0)
						      result-dim1
						      result-dim)))
				(with-facet (arr
					     ((data (apply
						     #'!faref
						     tensor
						     result-dim1))
					      'backing-array
					      :direction :input))
				  (declare (type (simple-array single-float) arr))
				  (loop for m fixnum upfrom 0 below iter-num
					do (case max-or-min
					     (:max
					      (cond
						((null m-val)
						 (setq m-val (aref arr m))
						 (setq m-pos (+ 0.0 m)))
						((> (aref arr m) m-val)
						 (setq m-val (aref arr m))
						 (setq m-pos (+ 0.0 m)))))
					     (:min
					      (cond
						((null m-val)
						 (setq m-val (aref arr m))
						 (setq m-pos (+ 0.0 m)))
						((< (aref arr m) m-val)
						 (setq m-val (aref arr m))
						 (setq m-pos (+ 0.0 m)))))
					     (T (error "!argmaxmin, max-or-min is :max or :min."))))
				  
				  (apply
				   #'(setf aref)
				   m-pos
				   return-array
				   result-dim)))
					;else			      
			      (apply-tensor (cdr rest-dims)
					    `(,@apply-dims ,i))))))
	(apply-tensor (!shape result)
		      nil)
	result))))



(defun !argmax (tensor &key (dim -1) (keepdims nil))
  "Returns the indices of the maximum value of all elements in the input tensor.

@begin(deflist)
@def(dim)
@term(The dimension to reduce. If nil, the argmax of the flattened input is returned.)
@def(keepdims)
@term(whether the output tensor has dim retained or not. Ignored if dim=-1)
@end(deflist)

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!randn `(5)))
;#Const((0.933... 0.158... 0.822... 0.881... 0.831...) :mgl t :shape (5))
(!argmax a)
;#Const((0.0) :mgl t :shape (1))
(setq a (!randn `(10 10 10)))
;#Const((((0.393... 0.658... ~ 0.003... 0.609...)         
;                   ...
;         (0.394... 0.252... ~ 0.688... 0.057...))        
;                 ...
;        ((0.325... 0.794... ~ 0.540... 0.381...)         
;                   ...
;         (0.310... 0.035... ~ 0.280... 0.431...))) :mgl t :shape (10 10 10))

(!argmax a :dim 2)

;#Const(((5.0 9.0 ~ 0.0 4.0)        
;                 ...
;        (2.0 0.0 ~ 2.0 5.0)) :mgl t :shape (10 10))

(!argmax a :dim 2 :keepdims t)
;#Const((((5.0 5.0 ~ 5.0 5.0)         
;                   ...
;         (4.0 4.0 ~ 4.0 4.0))        
;                 ...
;        ((2.0 2.0 ~ 2.0 2.0)         
;                   ...
;         (5.0 5.0 ~ 5.0 5.0))) :mgl t :shape (10 10 10))
@end[lang=lisp](code)
@end(section)"
  (if (null keepdims)
      (!argmaxmin tensor :max :dim dim)
      (!repeats (!unsqueeze (!argmaxmin tensor :max :dim dim) dim)
		dim
		(!shape tensor dim))))


(defun !argmin (tensor &key (dim -1) (keepdims nil))
  "Returns the indices of the minimum value of all elements in the input tensor.

@begin(deflist)
@def(dim)
@term(The dimension to reduce. If nil, the argmax of the flattened input is returned.)
@def(keepdims)
@term(whether the output tensor has dim retained or not. Ignored if dim=-1.)
@end(deflist)

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!randn `(5)))
;=>#Const((0.635... 0.101... 0.864... 0.563... 0.481...) :mgl t :shape (5))
(!argmin a)
;=>#Const((1.0) :mgl t :shape (1))

(setq a (!randn `(10 10 10)))
;#Const((((0.267... 0.113... ~ 0.142... 0.208...)         
;                   ...
;         (0.174... 0.948... ~ 0.232... 0.462...))        
;                 ...
;        ((0.454... 0.361... ~ 0.605... 0.731...)         
;                   ...
;         (0.099... 0.816... ~ 0.729... 0.996...))) :mgl t :shape (10 10 10))

(!argmin a)
;#Const((415.0...) :mgl t :shape (1))
@end[lang=lisp](code)
@end(section)"
  
  (if (null keepdims)
      (!argmaxmin tensor :min :dim dim)
      (!repeats (!unsqueeze (!argmaxmin tensor :min :dim dim) dim)
		dim
		(!shape tensor dim))))

