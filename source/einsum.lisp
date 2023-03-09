
(in-package :cl-waffe)

; Currently it won't work well... This is TODO


(defun get-sum-symbols (symbols)
  (let ((symbols (flatten symbols)))
    (map 'list
	 #'(lambda (x)
	     (setq symbols (delete x symbols :count 1)))
	 (remove-duplicates symbols))
    (remove-duplicates symbols)))

(defmacro -> (einsum &rest args)
  "do not use this."
  (declare (optimize (speed 3)))
  `(let ((einsum ,einsum)
	 ,@(map 'list #'(lambda (x) `(,x ,x)) args))
     (funcall einsum (list ,@args))))


(defmacro !einsum (&rest description)
  "do not use this."
  (declare (optimize (speed 3))
	   (type list description))
  (let* ((subscripts (loop for i fixnum upfrom 0 below (length `,description)
			   until (equal '-> (nth i `,description))
			   collect (nth i `,description)))
	 (explicts   (loop for i fixnum upfrom (1+ (position '-> `,description))
			   until (null (nth i `,description))
			   collect (nth i `,description)))
	 (iter-symbols (get-sum-symbols subscripts)))
    (declare (type list subscripts iter-symbols))
    (labels ((get-subscript-index (tensors symbol)
	       (declare (type list tensors)
			(type symbol symbol))
	       (loop named sloop
		     for i fixnum upfrom 0 below (length subscripts)
		     do (loop with ith-tensor = (nth i tensors)
			      for m fixnum
			      upfrom 0
				below (length (the list (nth i subscripts)))
			      do (let ((mth-symbol (nth m (nth i subscripts))))
				   (if (eql symbol mth-symbol)
				       (let ((size (!shape ith-tensor m)))
					 (declare (type fixnum size))
					 (return-from
					  sloop size)))))))
	     (get-subscript-index-iter (tensors symbol nth)
	       (declare (type symbol symbol))
	       (if (find symbol iter-symbols)
		   1
		   (or (get-subscript-index tensors symbol)
		       (shape-nth tensors nth))))
	     (shape-nth (tensors n)
	       (declare (type fixnum n))
	       (loop for i fixnum upfrom 0 below n
		     maximize (let ((res (!shape (nth i tensors) n)))
				(declare (type fixnum res))
				res)))
	     (parse-subscripts (n)
	       (nth n subscripts))
	     (parse-explicts (indices)
	       (map 'list #'(lambda (x)
			      (declare (type symbol x))
			      (if (find x iter-symbols)
					; Sum up about x
				  (nth (position x (the list (car explicts))) indices)
				  t))
		    (car explicts))))

      #'(lambda (tensors)
	  (declare (optimize (speed 3))
		   (type list tensors))
	  (let* ((result-dim (loop for m fixnum
				   upfrom 0
				     below (length
					    (the list (car explicts)))
				   collect (get-subscript-index-iter tensors (nth m (car explicts)) m)))
		 (result (!zeros result-dim)))
	    (labels ((sumup-next-iter (symbols &optional (indices nil))
		       (declare (optimize (speed 3)))
		       (loop with symbol = (car symbols)
			     for i fixnum
			     upfrom 0
			       below (get-subscript-index tensors symbol)
			     unless (null (cddr symbols)) ; remains > 2d
			       do  (sumup-next-iter
				    (cdr symbols)
				    `(,@indices ,i))
			     else
			       do (loop with indices = `(,@indices ,i)
					with tmp = nil
					for nth fixnum upfrom 0 below (length tensors)
					do (let* ((args-sub (parse-subscripts nth))
						  (exps-sub (parse-explicts args-sub))
						  (sumup-mode (= (the fixnum (apply #'* result-dim)) 1))
						  (value (apply
							  #'!aref
							  (nth nth tensors)
							  indices))
						  (init-it (= nth 0))
						  (transpose-point (loop for s fixnum upfrom 0 below (length (the list args-sub))
									 minimize (if (eql (the symbol (nth s args-sub)) (the symbol (nth s (car explicts))))
										      (1+ (length (the list args-sub)))
										      s)))
						  (transpose-point (if (= transpose-point (1+ (length (the list args-sub))))
								       nil
								       transpose-point)))
					    ; (print transpose-point)
					     ;(print args-sub)
					     ;(print exps-sub)
					     ;(print (car explicts))
					     ;(print tmp)
					     ;(print value)

					     (unless (null transpose-point)
					       (let ((shape (copy-list exps-sub)))
						 (setf (nth transpose-point shape) (!size value))
						 (setq value
						       (!reshape value shape))))

					     (setf (nth (case transpose-point
							  (0 1)
							  (1 0)
							  (T 0))
							exps-sub)
						   i)
					     (if init-it
						 (setq tmp value)
						 (setq tmp (!mul tmp value)))
					     (when (= nth (1- (length tensors))) ;reached an last term
					       (if sumup-mode
						   (setq result (!sum tmp))
						   (apply
						    #'(setf !aref)
						    tmp
						    result
						    exps-sub)
						  )))))))
	      (sumup-next-iter
	       (or iter-symbols
		   (car subscripts)))
	      result))))))

