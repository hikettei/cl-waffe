
(in-package :cl-waffe)

(define-condition Backend-Doesnt-Exists (simple-error)
  ((kernel-name :initarg :kernel)
   (node-name :initarg :node))
  (:report (lambda (c s)
	     (format s "The specified backend :~a doesn't have implementation for ~a."
		     (slot-value c 'kernel-name)
		     (slot-value c 'node-name)))))
