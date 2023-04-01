
(in-package :cl-waffe)

#|
The template of writing conditions:

[cl-waffe] Condition's name: content.

|#

(define-condition invaild-slot-name-waffe-object (simple-error)
  ((slot-name :initarg :name)
   (object-type :initarg :object-type))
  (:report (lambda (c s)
	     (format s "[cl-waffe] Invaild-slot-name-waffe-object: The name you specified ~a for defining ~a is invaild, since it was reserved by cl-waffe.~% Please consider using another naming."
		     (slot-value c 'slot-name)
		     (slot-value c 'object-type)))))

(defun invaild-slot-error (name model-type)
  (error (make-condition 'invaild-slot-name-waffe-object
			 :name name
			 :object-type model-type)))

#|
Shaping-Error
-> Aref-Shaping-Error
|#

(define-condition shaping-error (simple-error)
  ((excepted-shape :initarg :excepted)
   (wrong-shape    :initarg :butgot)
   (when-got-it    :initarg :when-to-got))
  (:report
   (lambda (c s)
     (format s "[cl-waffe] Shaping-Error: When operating ~a~%Excepted Shape: ~a~% But got: ~a~%"
	     (slot-value c 'when-got-it)
	     (slot-value c 'excepted-shape)
	     (slot-value c 'wrong-shape)))))

(defun throw-shaping-error (function-name excepted-shape wrong-shape)
  (error (make-condition 'shaping-error
			 :excepted excepted-shape
			 :butgot wrong-shape
			 :when-to-got function-name)))

(define-condition aref-shaping-error (shaping-error)
  ((content :initarg :content))
  (:report
   (lambda (c s)
     (format s "[cl-waffe] ~a" (slot-value c 'content)))))


(defmacro aref-shaping-error (content &rest args)
  `(error (make-condition 'aref-shaping-error
			  :content (format nil ,content ,@args))))
  

#|
Node-Error
-> ForwardNotNotFound
-> Backend-Not-Found
|#

(define-condition Backend-Doesnt-Exists (simple-error)
  ((kernel-name :initarg :kernel)
   (node-name :initarg :node))
  (:report (lambda (c s)
	     (format s "[cl-waffe] Backend-Doesnt-Exists: The specified backend :~a doesn't have implementation for ~a."
		     (slot-value c 'kernel-name)
		     (slot-value c 'node-name)))))
