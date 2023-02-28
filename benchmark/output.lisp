
(in-package :cl-waffe-benchmark)

; Here's utils for output the results.

(defvar *result*)

(defvar *speed-alert-min* 1.5)
(defvar *space-alert-min* 1.5)

(defun add-result (result)
  (push result *result*))

(defun parse-result (result)
  (let ((parsed-result (cl-ppcre:split " " (car result))))
    ; emmmm >< find another way, depending on SBCL 
    (values (read-from-string (nth 3 parsed-result))
	    (read-from-string (remove #\comma (nth (- (length parsed-result) 5) parsed-result))))))

;koreha kuso ko-do T_T
(defun save-result (stream)
  (let ((results (reverse *result*)))
    (dolist (result results)
      (let ((kernels (reverse result)))

	(let ((cl-waffe-result (car kernels))
	      (rest-result (cdr kernels)))
	  (multiple-value-bind (original-time original-space)
	      (parse-result (cdr cl-waffe-result))

	    (format stream "~%# ~a ~%~%### cl-waffe~%~%" (car cl-waffe-result))
	    (format stream "Time: ~as~%~%" original-time)
	    (format stream "Total Consed: ~aMB~%~%" (coerce (/ original-space 1e6) 'single-float))

	    (dolist (r rest-result)
	      (multiple-value-bind (r-time r-space)
		  (parse-result (cdr r))

		(format stream "~%### ~a ~%~%~a is:~%~%" (car r) (car r))

		(let ((speed-rate (if (= r-time 0.0)
				      -1.0
				      (coerce (/ original-time r-time) 'single-float)))
		      (space-rate (if (= r-space 0.0)
				      -1.0
				      (coerce (/ original-space r-space) 'single-float))))
		  (if (>= speed-rate *speed-alert-min*)
		      (format stream "❗️**Time: ~ax faster, (~as)**~%~%" speed-rate r-time)
		      (format stream "Time: ~ax faster, (~as)~%~%" speed-rate r-time))
		  (if (>= space-rate *space-alert-min*)
		      (format stream "❗️**Total Consed: ~ax smaller, (~aMB)**~%~%" space-rate (coerce (/ r-space 1e6) 'single-float))
		      (format stream "Total Consed: ~ax smaller, (~aMB)~%~%" space-rate (coerce (/ r-space 1e6) 'single-float))))))))))))

