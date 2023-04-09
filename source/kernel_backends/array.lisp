
(in-package :cl-waffe.kernel)


(defclass wvec (cube)
  ((ctype
    :initform *dtype*
    :initarg :ctype :reader vec-ctype)
   (initial-element
    :initform 0 :initarg :initial-element
    :reader vec-initial-element)
   (size :initarg :size :reader vec-size)
   ;; The number of bytes SIZE number of elements take.
   (n-bytes :reader vec-n-bytes)))

(defun dtype-size (ctype)
  (case ctype
    (:short 2)
    (:float 4)
    (:double 8)))

(defun dtype->lisp (dtype)
  (case dtype
    (:short 'short-float)
    (:float 'single-float)
    (:double 'double-float)))
			 
(defvar *foreign-pool* (make-instance 'mgl-mat::foreign-pool))

(defun alloc-static-vector (ctype length initial-element)
  (prog1
      (if initial-element
          (static-vectors:make-static-vector
           length :element-type (case ctype
				  (:short 'short-float)
				  (:float 'single-float)
				  (:double 'double-float))
		  :initial-element (coerce initial-element (dtype->lisp ctype)))
          (static-vectors:make-static-vector
           length :element-type (dtype->lisp ctype)))
    (mgl-mat::with-foreign-array-locked (*foreign-pool*)
      (incf (mgl-mat::n-static-arrays *foreign-pool*))
      (incf (mgl-mat::n-static-bytes-allocated *foreign-pool*)
            (* length (dtype-size ctype))))))

(defun coerce-to-dtype (element &key (ctype *dtype*))
  (case ctype
    (:short (coerce element 'short-float))
    (:float (coerce element 'single-float))
    (:double (coerce element 'double-float))))

(defmethod initialize-instance :after ((vec wvec) &key &allow-other-keys)
  (setf (slot-value vec 'n-bytes)
        (* (vec-size vec) (dtype-size (vec-ctype vec))))
  (when (vec-initial-element vec)
    (setf (slot-value vec 'initial-element)
          (coerce-to-dtype (vec-initial-element vec) :ctype (vec-ctype vec))))
  (mgl-mat::note-allocation (vec-n-bytes vec)))

(defun will-be-copied-over-p (vec)
  (find-if #'facet-up-to-date-p (facets vec)))

(defun vec-facet-to-char (vec facet)
  (let* ((name (facet-name facet))
         (char (aref (symbol-name name) 0)))
    (if (facet-up-to-date-p* vec name facet)
        (char-upcase char)
        (char-downcase char))))

(defun print-vec-facets (mat stream)
  (let ((chars (mapcar (lambda (facet)
                         (vec-facet-to-char mat facet))
                       (facets mat))))
    (if chars
        (format stream "~{~A~}" (sort chars #'char-lessp))
        (format stream "-"))))


(defmethod make-facet* ((vec wvec) (facet-name (eql 'lisp-vector)))
  (cond ((and (vec-initial-element vec)
              (not (will-be-copied-over-p vec)))
         (make-array (vec-size vec)
                     :element-type (case (vec-ctype vec)
				     (:short 'short-float)
				     (:float 'single-float)
				     (:double 'double-float))
                     :initial-element (vec-initial-element vec)))
        (t
         (make-array (vec-size vec)
                     :element-type (case (vec-ctype vec)
				     (:short 'short-float)
				     (:float 'single-float)
				     (:double 'double-float))))))

(defmethod make-facet* ((vec wvec) (facet-name (eql 'static-vector)))
  (let ((vector (alloc-static-vector (vec-ctype vec) (vec-size vec)
				     (if (will-be-copied-over-p vec)
					 nil
					 (vec-initial-element vec)))))
    (values vector nil t)))


(defmethod destroy-facet* ((facet-name (eql 'lisp-vector)) facet)
  (declare (ignore facet))
  ; deleted by gc.
  )

(defmethod destroy-facet* ((facet-name (eql 'static-vector)) facet)
  (mgl-mat::free-static-vector (facet-value facet)))

