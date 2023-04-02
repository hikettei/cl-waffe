
(in-package :cl-waffe)

#|
The template of writing conditions:

[cl-waffe] Condition's name: content.

|#

(define-condition invaild-slot-name-waffe-object (simple-error)
  ((slot-name :initarg :name)
   (object-type :initarg :object-type))
  (:documentation
   "Occurs if the arguments used when defining the object are reserved for cl-waffe.")
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
  (:documentation
   "In any operation of cl-waffe, If there's an error related to shaping, it occurs.")
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
  (:documentation
   "Occurs when the !aref/(setf !aref) argument is invalid and a copy could not be made.")
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

(define-condition node-error (simple-error)
  ((content :initarg :content))
  (:documentation
   "Errors related to defnode/defmodel.")
  (:report
   (lambda (c s)
     (format s "[cl-waffe] ~a" (slot-value c 'content)))))

(defmacro node-error (content &rest args)
  `(error (make-condition 'node-error
			  :content (format nil ,content ,@args))))


(define-condition Backend-Doesnt-Exists (node-error)
  ((kernel-name :initarg :kernel)
   (node-name :initarg :node))
  (:documentation
   "Occurs when the modified backend does not exist and *restart-non-exist-backend* is nil.")
  (:report (lambda (c s)
	     (format s "[cl-waffe] Backend-Doesnt-Exists: The specified backend :~a doesn't have implementation for ~a."
		     (slot-value c 'kernel-name)
		     (slot-value c 'node-name)))))

(define-condition backward-error (node-error)
  ((content :initarg :content))
  (:documentation
   "Occurs when can't backward due to some problems.")
  (:report
   (lambda (c s)
     (format s "[cl-waffe] ~a" (slot-value c 'content)))))

(defmacro backward-error (content &rest args)
  `(error (make-condition 'backward-error
			  :content (format nil ,content ,@args))))
