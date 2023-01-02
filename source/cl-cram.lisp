
(defpackage :cl-cram
  (:use :cl)
  (:export
   #:pdotimes
   #:pdolist
   #:pmap
   #:init-progress-bar
   #:update
   #:with-progress-bar
   #:discard-all-progress-bar
   #:*progress-bar-ascii*
   #:*blank*
   #:*progress-bar-enabled*))

(in-package :cl-cram)

(declaim (inline backward-lines))

(defparameter *indent* 0)
(defparameter *number-of-bar* 0)

(defparameter *progress-bar-enabled* t)
(defparameter *all-of-progress-bars* nil)

(defparameter *progress-bar-ascii* "█")
(defparameter *blank* "_")

(defstruct (progress-bar-status (:conc-name pbar-)
				(:predicate progress-bar))
  (total nil      :type fixnum)
  (count 0        :type fixnum)
  (desc "PROG"    :type string)
  (desc-len nil   :type fixnum)
  (start-time nil :type fixnum)
  (nth-bar nil    :type fixnum))

(defun backward-lines ()
  (write-char #\Return)
  (write-char #\Rubout))

(defmacro init-progress-bar (var desc total)
  `(progn
     (setq ,var (make-progress-bar-status :total ,total
					  :desc ,desc
					  :desc-len (length ,desc)
					  :start-time (get-universal-time)
					  :nth-bar *number-of-bar*))
     (incf *number-of-bar* 1)
     (setq *indent* (max (length ,desc) *indent*))
     (setq *all-of-progress-bars* (concatenate 'list *all-of-progress-bars*
					       (list ,var)))
     ,var))

(defmacro discard-all-progress-bar ()
  (defparameter *all-of-progress-bars* nil)
  (defparameter *number-of-bar* 0))

(defmacro progress-percent (status)
  `(fround (* 100 (/ (pbar-count ,status) (pbar-total ,status)))))

;(declaim (ftype (function (progress-bar-status fixnum)) update))
(defun update (status count &key desc reset)
  ;(declare (optimize (speed 3) (safety 0) (debug 0)))
  (incf (pbar-count status) count)
  (if reset
      (setf (pbar-count status) 0))
  (if desc
      (setf (pbar-desc status) desc))
  ;(if *progress-bar-enabled*
  ;    (backward-lines))
  (format t "~C" #\newline)
  (dolist (i *all-of-progress-bars*)
    (format t (render i)))
  nil)

;(declaim (ftype (function (progress-bar-status) string) render))
(defun render (status)
  ;(declare (optimize (speed 3) (safety 0) (debug 0)))
  (with-output-to-string (bar)
    (let ((spl (- *indent* (pbar-desc-len status) -1)))
      (write-string (pbar-desc status) bar)
      (dotimes (_ spl) (write-string " " bar))
      (write-string ":" bar))
    (let* ((n (round (progress-percent status)))
	   (r (round (if (>= (/ n 10) 10) 10 (/ n 10)))))
      (if (< n 100)
	  (write-string " " bar))
      (write-string (write-to-string n) bar)
      (write-string "% |" bar)
      (dotimes (_ r) (write-string *progress-bar-ascii* bar))
      (dotimes (_ (- 10 r)) (write-string *blank* bar)))
    (write-string "| " bar)
    (write-string (write-to-string (pbar-count status)) bar)
    (write-string "/" bar)
    (write-string (write-to-string (pbar-total status)) bar)
    (write-string " [" bar)
    (let* ((now-time (get-universal-time))
	   (dif (- now-time (pbar-start-time status))))
      (write-string (write-to-string dif) bar)
      (write-string "s] " bar))))

(defmacro with-progress-bar (var desc total &body body)
  `(progn
     (init-progress-bar ,var ,desc ,total)
     ,@body
     (discard-all-progress-bar)))


(defmacro pdotimes ((var count) &body body)
  (let ((bar (gensym))
	(r (gensym)))
    `(let ((,bar nil))
       (init-progress-bar ,bar "pdotimes" ,count)
       (let ((,r (dotimes (,var ,count)
		  (update ,bar 1)
		  ,@body)))
	 (princ #\newline)
	 (discard-all-progress-bar)
	 ,r))))

(defmacro pdolist ((var list) &body body)
  (let ((bar (gensym))
	(r (gensym)))
	`(let ((,bar nil))
	   (init-progress-bar ,bar "pdolist" (length ,list))
	   (let ((,r (dolist (,var ,list)
		      (update ,bar 1)
		      ,@body)))
	     (princ #\newline)
	     (discard-all-progress-bar)
	     ,r))))

(defmacro pmap (result-type function first-sequence)
  (let ((result (gensym))
	(bar (gensym)))
    `(let ((,bar nil))
       (init-progress-bar ,bar "pmap" (length ,first-sequence))
       (let ((,result (map ,result-type #'(lambda (x) (update ,bar 1)
					   (funcall ,function x))
			  ,first-sequence)))
	 (princ #\newline)
	 (discard-all-progress-bar)
	 ,result))))

