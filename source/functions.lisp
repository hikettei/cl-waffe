
(in-package :cl-waffe)

#|
Here's Mathematical Functions and Utils:
  1.Model-List
  2.!aref/(setf !aref)
|#

(defun !softmax-function (x &key (avoid-overflow t))
  "Applying softmax.

!softmax has three behaivour depending on the number of dimensions."
  (declare (optimize (speed 3))
	   (type waffetensor x))
  (case (!dims x)
    (1 (!softmax-function (!unsqueeze x)))
    (2 (let* ((x1 (if avoid-overflow
		      (!sub x (!mean x 1))
		      x))
	      (z (!sum (!exp x1) 1)))
	 (!!div (!exp x1) z)))
    (3 (let* ((xs (!split x 1 :axis 0)))
	 ; Todo: Make here destructive.
	 (apply
	  #'!concatenate
	  0
	  (map 'list #'(lambda (tensor)
			 (!unsqueeze
			  (!softmax-function
			   (!squeeze tensor 0)
			   :avoid-overflow avoid-overflow)))
	       xs))))
    (T (error "!softmax: Not implemented. softmax only supports where (!dims tensor) <= 3."))))

(defmodel SoftMaxNode (avoid-overflow)
  :parameters ((avoid-overflow avoid-overflow))
  :forward ((x)
	    (!softmax-function x :avoid-overflow (self avoid-overflow))))

(defun !softmax (x &key (avoid-overflow t))
  "Applying softmax to x. !softmax has three behaviours depending on the number of dimensions.

The number of dims is...
@begin(deflist)
@def(1)
@begin(term)
Softmax is applied to dim=0
@begin[lang=lisp](code)
(setq a (!randn `(10)))
(!softmax a)
;#Const((0.910... 0.886... ~ 0.802... 0.616...) :mgl t :shape (10))
@end[lang=lisp](code)
@end(term)

@def(2)
@begin(term)
Softmax is applied to dim=0
@begin[lang=lisp](code)
(setq a (!randn `(10 10)))
;#Const(((-0.29... -1.99... ~ -0.36... 1.725...)        
;                 ...
;        (0.695... -0.94... ~ 1.179... 0.655...)) :mgl t :shape (10 10))

(!softmax a)
;#Const(((0.064... 0.011... ~ 0.060... 0.489...)        
;                 ...
;        (0.129... 0.024... ~ 0.209... 0.124...)) :mgl t :shape (10 10))
@end[lang=lisp](code)
@end(term)

@def(3)
@begin(term)
Softmax is applied to dim=0
@begin[lang=lisp](code)
(setq a (!randn `(10 10 10)))
;#Const((((2.585... 0.517... ~ 0.428... 0.059...)         
;                   ...
;         (-2.11... 0.308... ~ -0.91... 0.649...))        
;                 ...
;        ((-0.75... 1.030... ~ 0.656... -0.00...)         
;                   ...
;         (-0.37... -0.52... ~ 1.589... -0.10...))) :mgl t :shape (10 10 10))

(!softmax a)
;#Const((((0.374... 0.047... ~ 0.043... 0.029...)         
;                   ...
;         (0.010... 0.115... ~ 0.033... 0.162...))        
;                 ...
;        ((0.029... 0.172... ~ 0.118... 0.061...)         
;                   ...
;         (0.048... 0.041... ~ 0.345... 0.063...))) :mgl t :shape (10 10 10))
@end[lang=lisp](code)
@end(term)

@def(4)
@begin(term)
Todo: currently, it returns error.
@begin[lang=lisp](code)
@end[lang=lisp](code)
@end(term)
@end(deflist)"
  (declare (type waffetensor x))
  (call (SoftMaxNode avoid-overflow) x))

; Model Lists

(defmodel model-list (model-list)
  :document (with-usage "model-list"
	      :overview "define model sequentially, (e.g. x = (sequence `((layer1) (layer2))), (call x 1 tensor) => layer1's output)"
	      :args "model1 model2 ..."
	      :forward "@cl:param(index) represents the index of models. @cl:param(args) is the arguments for index-th model."
	      :step-args "index &rest args")
  :parameters ((mlist model-list))
  :forward ((index &rest args)
	    (error "model-list couldn't pass call correctly")))

(defun mlist (&rest models)
  "define mlist"
  (model-list models))

(defun mth (index mlist)
  "Accessor for model-list"
  (declare (type model-list mlist))
  (nth (typecase index
	 (waffetensor (data index))
	 (T index))
       (model-list-mlist mlist)))


