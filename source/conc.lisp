
(in-package :cl-waffe)

(defnode ConcatenateTensorNode (axis)
  :optimize t
  :parameters ((axis axis :type fixnum)
	       (shape t))
  :forward ((&rest tensors)
	    (let ((first-shape (the list (!shape (car tensors)))))
	      (loop for i fixnum upfrom 0 below (length tensors)
		    unless (equal first-shape (the list (!shape (nth i tensors))))
		      do (error "cl-waffe.stack!: all tensors must be consisted of the same shapes, but got ~a. excepted:~a" (nth i tensors) first-shape))
	      (let* ((result-shape
		       (loop for i fixnum
			     upfrom 0
			       below (length first-shape)
			     if (= i (self axis))
			       collect (the fixnum (* (the fixnum (length tensors)) (the fixnum (nth i first-shape))))
			     else
			       collect (nth i first-shape)))
		     (result (!zeros result-shape)))
		(setf (self shape) (!shape (car tensors) (self axis)))
		(stack! (self axis)
			(map 'list #'value tensors)
			(data result))
		result)))
  :backward ((dy)
	     (!split dy (self shape) :axis (self axis))))

(defnode SplitTensorNode (split-size axis)
  :parameters ((split-size split-size :type fixnum)
	       (axis axis :type fixnum)
	       (grads nil)
	       (prev-shape t))
  :forward ((tensor)
	    (setf (self prev-shape) (!shape tensor))
	    (setf (self grads) nil)
	    (loop
	      with each-tensor-shape = (let ((shape (copy-list (!shape tensor))))
					 (setf (nth (self axis) shape) (self split-size))
					 shape)
	      with tmp-areas = (loop for i fixnum upfrom 0 below (self axis)
				     collect t)
	      for ith fixnum
	      upfrom 0
		below (the fixnum
			   (multiple-value-bind (k rest)
			       (round
				(/
				 (!shape tensor (self axis))
				 (self split-size)))
			     
			     (declare (type fixnum k))
			     (if (> rest 0.0)
				 (1+ k)
				 k)))
	      collect (let ((start-index (* ith (self split-size)))
			    (end-index (* (1+ ith) (self split-size))))

			(if (<= end-index (!shape tensor (self axis)))
			    (apply
			     #'%saref
			     nil
			     tensor
			     `(,@tmp-areas (,start-index
					    ,end-index)))
			    (let ((res (!zeros each-tensor-shape)))
			      (copy!
			       (data
				(apply
				 #'%saref
				 nil
				 tensor
				 `(,@tmp-areas
				   (,start-index
				    ,(!shape tensor (self axis))))))
			       (data res))
			      res)))))
  :backward ((dy)
	     (error "SplitNodeTensor: Currently backward is undefined because there's a room to optimize.")
	     (list (const nil))))

(defun !concatenate (axis &rest tensors)
  "Concatenates the given sequence of @cl:param(tensors) in the given @cl:param(axis). All tensors must have the same shape.

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!randn `(3 3 3)))
;#Const((((1.000... -0.00... -0.25...)         
;                   ...
;         (1.473... -0.44... 1.680...))        
;                 ...
;        ((0.569... 0.852... 0.405...)         
;                   ...
;         (0.024... 0.756... 0.383...))) :mgl t :shape (3 3 3))

(!concatenate 0 a a a)
;#Const((((1.000... -0.00... -0.25...)         
;                   ...
;         (1.473... -0.44... 1.680...))        
;                 ...
;        ((0.569... 0.852... 0.405...)         
;                   ...
;         (0.024... 0.756... 0.383...))) :mgl t :shape (9 3 3))

(mgl-mat:M= (data (!aref * '(0 3)))
            (data (!aref * '(3 6))))
;T
@end[lang=lisp](code)
@end(section)"
  (declare (optimize (speed 3))
	   (type fixnum axis))
  (apply #'call (ConcatenateTensorNode axis) tensors))

(defun !stack (axis &rest tensors)
  "Stacks the given @cl:param(tensors) in the specified @cl:param(axis).

Internally, !stack @b(adds 1 to the specified axis) before calling !concatenate.

Note: Currently, when unsqueezing given tensors, !stack creates copies every time in order to prevent side effects. To avoid this, !concatenate is recommended to use. @b((TO FIX))

@begin(section)
@title(Example)
@begin[lang=lisp](code)
(setq a (!randn `(2 2 2)))

;#Const((((-0.83... -1.74...)
;         (0.119... 0.162...))
;        ((-1.81... 0.907...)
;         (-0.50... -0.96...))) :mgl t :shape (2 2 2))

(!stack 0 a a a)
;#Const(((((-0.83... -1.74...)
;          (0.119... 0.162...))
;         ((-1.81... 0.907...)
;          (-0.50... -0.96...)))        
;                 ...
;        (((-0.83... -1.74...)
;          (0.119... 0.162...))
;         ((-1.81... 0.907...)
;          (-0.50... -0.96...)))) :mgl t :shape (3 2 2 2))

(mgl-mat:M= (data (!aref * 0)) (data (!aref * 1)))
; T
@end[lang=lisp](code)
@end(section)"
  (let ((tensors (map 'list
		      #'(lambda (tensor)
			  ; bug: (!stack 0 a a a) <- reshaped for multiple times.
			  (!disallow-destruct tensor)
			  (!unsqueeze tensor axis))
		      tensors)))
    (apply #'call (ConcatenateTensorNode axis) tensors)))

(defun !split (tensor split-size &key (axis 0))
  "Splits the tensor into chunks in the specified @cl:param(axis). Each chunk is a copy of original tensor.

split-size indicates the strides of each chunk, that is, @cl:param(tensor) will be split into equalliy size of @cl:param(split-size).

split-size must be fixnum.

Note: currently !split's backward returns error. this is because there's a room to optimize backward and until optimized them i wont make it.

Alternatively, !aref, (setf !aref) is available.

@begin(section)
@title(Example)
@begin[lang=lisp](code)

(setq a (!randn `(4 2 2)))
;#Const((((-0.48... -1.22...)
;         (0.251... 0.476...))        
;                 ...
;        ((-0.66... 1.045...)
;         (-0.44... 1.592...))) :mgl t :shape (4 2 2))

(!split a 2)
;(#Const((((-0.48... -1.22...)
;         (0.251... 0.476...))
;        ((0.864... -0.93...)
;         (-0.43... 0.346...))) :mgl t :shape (2 2 2))
; #Const((((-1.91... -0.63...)
;         (-0.08... 0.867...))
;        ((-0.66... 1.045...)
;         (-0.44... 1.592...))) :mgl t :shape (2 2 2)))

; the rests are filled with 0.0
(!split a 3)
;(#Const((((-0.48... -1.22...)
;         (0.251... 0.476...))        
;                 ...
;        ((-1.91... -0.63...)
;         (-0.08... 0.867...))) :mgl t :shape (3 2 2))
; #Const((((-0.66... 1.045...)
;         (-0.44... 1.592...))        
;                 ...
;        ((0.0 0.0)
;         (0.0 0.0))) :mgl t :shape (3 2 2)))
@end[lang=lisp](code)
@end(section)"
  (declare (type waffetensor tensor)
	   (type fixnum split-size axis)
	   (optimize (speed 3)))
  (call (SplitTensorNode split-size axis) tensor))

(defmacro !vstack (&rest tensors)
  "!vstack is the equivalent to !concatenate(axis=0)"
  `(!concatenate 0 ,@tensors))

(defmacro !hstack (&rest tensors)
  "!vstack is the equivalent to !concatenate(axis=1)"
  `(!concatenate 1 ,@tensors))

