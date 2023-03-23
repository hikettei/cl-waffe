
(in-package :cl-waffe)

(define-condition Backend-Doesnt-Exists (simple-error)
  ((kernel-name :initarg :kernel)
   (node-name :initarg :node))
  (:report (lambda (c s)
	     (format s "The specified backend :~a doesn't have implementation for ~a."
		     (slot-value c 'kernel-name)
		     (slot-value c 'node-name)))))

(define-condition invaild-slot-name-waffe-object (simple-error)
  ((slot-name :initarg :name)
   (object-type :initarg :object-type))
  (:report (lambda (c s)
	     (format s "Invaild parameter name: ~a for defining ~a, since it was reserved by cl-waffe.~% Please consider using another naming."
		     (slot-value c 'slot-name)
		     (slot-value c 'object-type)))))

(defun invaild-slot-error (name model-type)
  (error (make-condition 'invaild-slot-name-waffe-object
			 :name name
			 :object-type model-type)))
