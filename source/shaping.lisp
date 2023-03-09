
(in-package :cl-waffe)

(defnode ReshapeTensor (shape)
  :optimize t
  :parameters ((prev-shape T) (shape shape))
  :forward ((x) (setf (self prev-shape) (!shape x))
		(with-searching-calc-node :reshape x (self shape)))
  :backward ((dy)
	     (list (!reshape dy (self prev-shape)))))

(defun !reshape (x dim)
  "Return a new sysconst with changing its shape. x won't be modified.

If dims has the element of @cl:param(t), t is automatically inferred from the remaining dimensions and the number of elements in dim. (count t dim) must be 1 (Todo: Fix).

The total size of tensor must not be changed before or after the call to reshape.

See also: nil

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!randn `(10 10 10)))
(!reshape a '(1 10 100))
;#Const((((0.454... 0.277... ~ 0.536... 0.135...)         
;                   ...
;         (0.857... 0.714... ~ 0.169... 0.279...))) :mgl t :shape (1 10 100))

(!reshape a '(1 1 t))
;#Const((((0.454... 0.277... ~ 0.169... 0.279...))) :mgl t :shape (1 1 1000))
@end[lang=lisp](code)
@end(section)"
  (declare (type cons dim))
  (if (find t dim)
      (progn
	(unless (= (count t dim) 1)
	  (error "cl-waffe:!reshape: auto inference of shape supports only when (count t dim) = 1"))
	(let* ((dim (copy-list dim))
	       (total-size  (apply #'* (!shape x)))
	       (remain-size (apply #'* (map 'list (lambda (x)
						    (if (eql x T)
							1
							x))
					    dim)))
	       (predicted-dim (/ total-size remain-size)))
	  (setf (nth (position t dim) dim) predicted-dim)
	  (call (ReshapeTensor (assure-tensor dim)) (assure-tensor x))))
      (call (ReshapeTensor (assure-tensor dim)) (assure-tensor x))))


(defnode RepeatTensor (axis repeats)
  :optimize t
  :parameters ((axis axis) (repeats repeats))
  :forward ((x)
	    (with-searching-calc-node :repeat x (self axis) (self repeats)))
  :backward ((dy) (list (!sum dy (self axis)))))

(defun !repeats (x axis repeats)
  "Repeats @cl:param(x) along specified @cl:param(axis) by @cl:param(repeats), creating new sysconst.

x can be: mat or tensor.

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!randn '(1 3 3)))
;#Const((((0.333... 0.914... 0.260...)         
;                   ...
;         (0.611... 0.110... 0.113...))) :mgl t :shape (1 3 3))
(!repeats a 0 3)
;#Const((((0.333... 0.914... 0.260...)         
;                   ...
;         (0.611... 0.110... 0.113...))
;                 ...
;        ((0.333... 0.914... 0.260...)         
;                   ...
;         (0.611... 0.110... 0.113...))) :mgl t :shape (3 3 3))

(!repeats (const 10.0) 3 10)
;#Const(((((10.0 10.0 ~ 10.0 10.0)))) :mgl t :shape (1 1 1 10))
@end[lang=lisp](code)
@end(section)"
  (declare (type waffetensor x))
  (call (RepeatTensor (assure-tensor axis) (assure-tensor repeats)) (assure-tensor x)))

(defun !expands (tensor &rest expand-times)
  "todo: !expands hehave like pytorch's one.
(!expands tensor 3 3 3)...
implement with:... !sum with multiple axis"
  (declare (type waffetensor tensor)
	   (ignore tensor expand-times))

  ;(unless (= (!dims tensor) (length expand-times))
  ;  (error "!expands: ~a and ~a aren't compatiable. Their dims and length must be the same" tensor expand-times))

  (error "not implemented"))


(defun !unsqueeze (x &optional (dim 0) (count 1))
  "Returns a new tensor with a dimension of size one inserted at the specified position.

dim indicates the position, when dim=-1, it indicates a last dimension of @cl:param(x).

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!randn `(10 10)))
;#Const(((0.685... 0.827... ~ 0.076... 0.102...)        
;                 ...
;        (0.802... 0.571... ~ 0.207... 0.283...)) :mgl t :shape (10 10))
(!unsqueeze a)
;#Const((((0.685... 0.827... ~ 0.076... 0.102...)         
;                   ...
;         (0.802... 0.571... ~ 0.207... 0.283...))) :mgl t :shape (1 10 10))

(!unsqueeze a -1)
;#Const((((0.685...)         
;                   ...
;         (0.102...))        
;                 ...
;        ((0.802...)         
;                   ...
;         (0.283...))) :mgl t :shape (10 10 1))

(!unsqueeze a 2)
;#Const(((0.685... 0.827... ~ 0.076... 0.102...)        
;                 ...
;        (0.802... 0.571... ~ 0.207... 0.283...)) :mgl t :shape (10 10 1 1))
@end[lang=lisp](code)
@end(section)"
					; display error when (!dims x) >= dim
  (let ((s (copy-list (!shape x))))
    (dotimes (_ count)
      (case dim
	(0  (setq s `(1 ,@s)))
	(-1 (push 1 (cdr (nthcdr (1- (length s)) s))))
	(T  (push 1 (cdr (nthcdr (1- dim) s))))))
    (!reshape x s)))

(defun !squeeze (x &optional (dim nil))
  "Returns a new tensor with a dimension of size one removed at the specified position.

When dim=nil or -1, the last position of dim will be removed.

If the specified position of a tensor isn't one, !squeeze is skipped.

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!randn `(10 1 10)))
;#Const((((0.928... 0.556... ~ 0.697... 0.973...))        
;                 ...
;        ((0.368... 0.995... ~ 0.589... 0.716...))) :mgl t :shape (10 1 10))

(!squeeze a 1)
;#Const(((0.928... 0.556... ~ 0.697... 0.973...)        
;                 ...
;        (0.368... 0.995... ~ 0.589... 0.716...)) :mgl t :shape (10 10))

(!squeeze a -1)
;#Const((((0.928... 0.556... ~ 0.697... 0.973...))        
;                 ...
;        ((0.368... 0.995... ~ 0.589... 0.716...))) :mgl t :shape (10 1 10))

(setq a (!randn `(10 10 1)))
;#Const(((0.991... 0.248... ~ 0.610... 0.289...)        
;                 ...
;        (0.593... 0.177... ~ 0.374... 0.668...)) :mgl t :shape (10 10))
@end[lang=lisp](code)
@end(section)"
  (labels ((remove-nth (nth list)
	     (loop for i in list
		   for idx from 0
		   unless (= idx nth)
		     collect i)))
    (let ((s (!shape x)))
      (cond
	((null dim) (setq s (remove 1 s)))
	((eq dim 0) (setq s (if (= (car s) 1)
				(cdr s)
				s)))
	((eq dim -1) (setq s (if (= (car (last s)) 1)
				 (butlast s)
				 s)))
	(T (setq s (if (= (nth dim s) 1)
		       (remove-nth dim s)
		       s))))
      (!reshape x s))))

(defun !ravel () "Todo")
(defun !flatten (tensor)
  "Flattens input by reshaping it into a one-dimensional tensor.

The operation is the same as @c((!reshape tensor '(t)))

Example:
@begin[lang=lisp](code)

(setq a (!randn `(10 10)))
;#Const(((0.688... 0.580... ~ 0.013... 0.461...)        
;                 ...
;        (0.214... 0.248... ~ 0.540... 0.416...)) :mgl t :shape (10 10))

(!flatten a)
;#Const((0.688... 0.580... ~ 0.540... 0.416...) :mgl t :shape (100))
@end[lang=lisp](code)"
  (!reshape tensor '(t)))


