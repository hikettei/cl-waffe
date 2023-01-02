
(in-package :cl-termgraph)


(defparameter *dif* 1/16)
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
     (loop for i from s to (1- e) by *dif*
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
       ((and (< 0 ti) (< ti 0.5)) (second *positive-lines*))
       ((and (<= 0.5 ti) (< ti 1.0)) (third *positive-lines*))
       ((and (<= 1.0 ti) (< ti 1.5))  (fourth *positive-lines*))
       ((and (<= 1.5 ti) (< ti 3.0))  (fifth *positive-lines*))
       ((and (<= 3.0 ti) (< ti 4.5))  (sixth *positive-lines*))
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


(defun render (frame pallet)
  (with-output-to-string (graph)
    (loop for y from 1 to (slot-value frame 'height)
	  do (loop for x from 0 to (1- (slot-value frame 'width))
		   do (write-string (aref pallet x (- (slot-value frame 'height) y)) graph))
	     (write-char #\Newline graph))))

(defun make-listplot-frame (x y)
  (make-array `(,x ,y) :initial-element " "))

(defun mean (list)
  (if (= (length list) 0)
      0
      (/ (apply #'+ list) (length list))))

(defun pick-color (line color)
  (case color
    (:black
     (black line))
    (:red
     (red line))
    (:green
     (green line))
    (:blue
     (blue line))
    (:yellow
     (yellow line))
    (:magenta
     (magenta line))
    (:white
     (white line))
    (:cyan
     (cyan line))
    (T (error "No such color: ~a" color))))

(defun eq-ntimes (width &optional (word "="))
  (with-output-to-string (str) (dotimes (_ (* 2 width)) (format str word))))

(defun format-title (title start-from width word &optional default-base)
  (let ((base (if default-base default-base (eq-ntimes width word))))
    (setf (subseq base start-from (+ start-from (length title))) title)
    base))

(defun init-line (frame color)
  (let ((x (first (array-dimensions frame))))
    (loop for i from 0 to (1- x)
	  do (setf (aref frame i 0) (pick-color "–" color)))))

(defun repeat0-ntimes (n)
  (let ((a `(0)))
    (dotimes (_ (1- n))
      (push 0 a))
    a))

; 共通のheightでlistの値をnormalizeしておくこと
(defun listplot-write (frame list &optional (color :blue))
  (let* ((x (first  (array-dimensions frame)))
	 ;(y (second (array-dimensions frame)))
	 (list-width (length list))
	 (list (if (< (length list) x) (concatenate 'list list (repeat0-ntimes (- x (length list)))) list))
	 (points-per-1frame (/ (max x list-width) (min x list-width)))
	 (plot-points (loop for i from 0 to (1- list-width) by points-per-1frame
			    collect (mean (loop for l from (1+ i) to (+ i points-per-1frame) by 1 collect (nth (1- (round l)) list)))))
	 (plot-points-tmp (concatenate 'list plot-points `(0 0 0))))
    
    (dotimes (i (- (length plot-points) 1))
      (let ((point-y (nth i plot-points-tmp))
	    (line (case i
		    (0 (choose-line (cons 0 (car plot-points-tmp))
				    (cons 1 (second plot-points-tmp))
				    (cons 2 (third plot-points-tmp))))
		    
		    (T (choose-line (cons (1- i) (nth (1- i) plot-points-tmp))
				    (cons i      (nth i plot-points-tmp))
				    (cons (1+ i) (nth (1+ i) plot-points-tmp)))))))
	(setf (aref frame (min i (1- x)) (round point-y)) (pick-color line color))))))

(defun listplot-print (frame &key (x-label "x") (y-label "y") (descriptions) (title "No title:") (stream t))
  ; descriptions an list of `(:color "name" max min)
  (declare (ignore stream))
  (let ((width-len (car (array-dimensions frame))))
    (let ((graph (with-output-to-string (result)
		   (if title
		       (format result "||~a||~C~a~C" title #\newline y-label #\newline)
		       (format result "~a~C" y-label #\newline))
		   ;(if y-max
		   ;    (format result "~a_~C" y-max #\newline))
		   (loop for y from 1 to (second (array-dimensions frame))
			 do (progn
			      (loop for x from 0 to (1- (car (array-dimensions frame)))
				    do (progn
					 (write-string (aref frame x (- (second (array-dimensions frame)) y)) result)))
			      (write-char #\newline result)))
		   (write-string (format-title x-label (* 1 (- width-len (1+ (length x-label)))) width-len " "
					       (format-title "0" 0 width-len " ")) result)
		   (write-char #\newline result)
		   (if descriptions
		       (let ((max-desc-title (apply #'max (map 'list #'(lambda (desc) (length (second desc))) descriptions))))
			 (dolist (desc descriptions)
			   (write-string (pick-color (format nil "|~a: (~a ... ~a)~C"
						 (format-title (second desc) 1 width-len "" (eq-ntimes max-desc-title " "))
						 (third desc)
						 (fourth desc)
						 #\newline) (car desc))
					 result)))))))
      (princ graph))))
