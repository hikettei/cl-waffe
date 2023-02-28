
(in-package :cl-waffe-benchmark)

; Here's utils for output the results.

(defun save-result (stream)
  (format stream "~a" *result*))

