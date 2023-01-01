
(in-package :cl-termgraph)


(defparameter *dif* 1/3)
(defparameter *positive-lines* `("⠒⠒" "⣠⣤" "⣠⣰" "⣠⣼" "⡜⡜" "⣼⡜" "⡇⡇"))
(defparameter *negative-lines* `("⠒⠒" "⣤⣄" "⣆⣄" "⣧⣄" "⢣⢣" "⢣⣧" "⡇⡇"))

(defgeneric plot (frame pallet))
(defmacro mbind (&rest args)
  `(multiple-value-bind ,@args))

(defclass figure-graph-frame (simple-graph-frame)
  ((figure :accessor figure
	   :initarg :figure
	   :initform nil)
   (from :accessor figurefrom
	 :initarg :from
	 :initform nil)
   (end :accessor figureend
	:initarg :end
	:initform nil)
   (name :accessor name
	 :initarg :name
	 :initform nil)))

(defclass parameter-graph-frame (figure-graph-frame)
  ((parameter :accessor parameter
	      :initarg :parameter
	      :initform nil)
   (by :accessor move-by
       :initarg :by
       :initform 1)
   (pstart :accessor pstart
	   :initarg :pstart
	   :initform -3)
   (pend :accessor pend
	 :initarg :pend
	 :initform 3)))

(defmethod collect-figure-points ((frame figure-graph-frame))
  (with-slots ((figure figure) (s from) (e end)) frame
     (loop for i from s to e by *dif*
	  collect (handler-case (funcall figure i)
		    (error (x) ; set as undefined
		      (declare (ignore x))
			       `(,i . nil))
		    (:no-error (x) `(,i . ,x))))))

(defun max-points (points)
  (loop for i in (map 'list #'(lambda (p) (cdr p)) points)
	maximize i))

(defun min-points (points)
  (loop for i in (map 'list #'(lambda (p) (cdr p)) points)
	minimize i))

(defun maxmin-points (points)
  (values (max-points points) (min-points points)))

(defmacro choose-line (p1 p2 p3 &optional (opt 0))
  `(let ((ti (+ ,opt (3p-tilt-ave ,p1 ,p2 ,p3))))
     (cond
       ((= ti 0) (first *positive-lines*))
       ((and (< 0 ti) (< ti 0.5))    (second *positive-lines*))
       ((and (<= 0.5 ti) (< ti 1.0)) (third *positive-lines*))
       ((and (<= 1.0 ti) (< ti 1.5))  (fourth *positive-lines*))
       ((and (<= 1.5 ti) (< ti 3.0))  (fifth *positive-lines*))
       ((and (<= 3.0 ti) (< ti 4.5))    (sixth *positive-lines*))
       ((>= ti 4.5) (seventh *positive-lines*))
       ((and (< -0.5 ti) (<= ti 0)) (second *negative-lines*))
       ((and (< -1.0 ti) (<= ti -0.5)) (third *negative-lines*))
       ((and (< -1.5 ti) (<= ti -1.0)) (fourth *negative-lines*))
       ((and (< -3.0 ti) (<= ti -1.5)) (fifth *negative-lines*))
       ((and (< -4.5 ti) (<= ti -3.0)) (sixth *negative-lines*))
       ((<= ti -4.5) (seventh *negative-lines*)))))

(defmacro tilt (p1 p2)
  `(if (and ,p1 ,p2)
       (/ (- (cdr ,p1) (cdr ,p2)) (- (car ,p1) (car ,p2)))
       0))

(defmacro 3p-tilt-ave (p1 p2 p3)
  `(/ (+ (tilt ,p1 ,p2) (tilt ,p2 ,p3)) 2))


(defmethod plot ((frame figure-graph-frame) (pallet (eql nil)))
  (plot frame (draw-graph-base frame)))

(defmethod plot ((frame figure-graph-frame) pallet)
  (with-slots ((s from) (e end) (x width) (y height)) frame
    (let* ((points (collect-figure-points frame))
	   (figure-size (+ (abs e) (abs s)))
	   (expa-rate-x (if (= figure-size 0) 1 (/ x figure-size)))
	   (expa-rate-y (/ y (mbind (max min) (maxmin-points points)
			       (let ((sum (+ (abs max) (abs min))))
				 (if (= sum 0) y sum)))))
	   (expa-rate (- (max expa-rate-x expa-rate-y) 2))
	   (expaed-points (map 'list #'(lambda (p)
					 `(,(* (car p) expa-rate-x) .
					   ,(* (cdr p) 1))) points))
	   (xmin-abs (abs (round (caar expaed-points))))
	   (ymin-abs (abs (round (min-points expaed-points)))))

      (loop for i from 0 to (1- x)
	    do (setf
		(aref pallet i (round (funcall (slot-value frame 'figure) 0)))
		"  "))
      
      (loop for i from 0 to (1- (length expaed-points))
	    do (let* ((p1 (if (= i 0 ) nil (nth (1- i) expaed-points)))
	 	      (p2 (nth i expaed-points))
		      (p3 (nth (1+ i) expaed-points))
		      (next-line (choose-line p1 p2 p3))
		      (x (round (car p2)))
		      (y (round (cdr p2)))
		      (ave (3p-tilt-ave p1 p2 p3)))
		 (setf (aref pallet
			     (+ x xmin-abs)
			     (+ y ymin-abs))
		       (blue next-line))
		 (if (and (or (> ave 1.5)
			      (< ave -1.5))
			  (equal (aref pallet
			     (+ x xmin-abs)
			     (+ y ymin-abs -1)) "  "))
		     (setf (aref pallet
			     (+ x xmin-abs)
			     (+ y ymin-abs -1))
			   (blue (choose-line p1 p2 p3 (if (> ave 0)
							   1.51 -1.51)))))))
	  
      (princ (render frame pallet)))) nil)

(defmethod plot ((frame parameter-graph-frame) pallet)
  (with-slots ((width width)
	       (height height)
	       (figure figure)
	       (from from)
	       (end end)
	       (name name)
	       (parameter parameter)
	       (pstart pstart)
	       (pend pend)
	       (by by)) frame
    (macrolet ((let-parameter (p value f)
		 ; f(x,a) => f(x)
		 `(lambda (@) (funcall ,f @ ,value))))
      (loop for i from pstart to pend by by
	    do (fresh-line)
	       (format t "|====|Plotting ~a=~a |=====|" parameter i)
	       (fresh-line)
	       (plot (make-instance 'figure-graph-frame
				    :from from
				    :end end
				    :name name
				    :width width
				    :height height
				    :figure (let-parameter parameter i figure)
				    :from from
				    :end end) pallet)
	       (sleep (/ by 2))))))

(defun render (frame pallet)
  (with-output-to-string (graph)
    (loop for y from 0 to (slot-value frame 'height)
	  do (loop for x from 0 to (slot-value frame 'width)
		   do (write-string (aref pallet x (- (slot-value frame 'height) y)) graph))
	     (write-char #\Newline graph))))

