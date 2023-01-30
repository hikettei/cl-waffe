
(defpackage :cl-waffe.caches
  (:use :cl :cl-waffe :mgl-mat)
  (:export
   #:with-cache
   #:return-caches
   #:free-cache
   #:caches-gc))

; This package exports features for making caches (sysconst)
; Tracking sysconst's usage every step, cl-waffe optimizes the number of cl-waffe's making copy.

(in-package :cl-waffe.caches)

; todo: add depends on tg, bordeaux-threads
(defvar *thread-caches*
  (tg:make-weak-hash-table :weakness :key))
(defvar *thread-cache-lock*
  (bordeaux-threads:make-lock "thread cache lock"))

; Todo multi thread safe
(defvar *thread-callns*
  (tg:make-weak-hash-table :weakness :key))

(defstruct CacheData
  (calln 0 :type fixnum)
  (calln-per-step 0 :type fixnum)
  (lock nil :type boolean))

(defun update-calln (idx)
  (let ((calln (or (gethash idx *thread-callns*)
		   (make-cachedata :calln -1 :calln-per-step -1))))
    (unless (cachedata-lock calln)
      (incf (cachedata-calln-per-step calln) 1))
    
    (incf (cachedata-calln calln) 1)

    (when (cachedata-lock calln)
      (setf (cachedata-calln calln)
	    (case (cachedata-calln-per-step calln)
	      (0 0)
	      (T (mod (cachedata-calln calln)
		      (cachedata-calln-per-step calln))))))
    
    (setf (gethash idx *thread-callns*) calln)))

(defun lock-calln (idx)
  (let ((calln (gethash idx *thread-callns*)))
    (when calln
      (setf (cachedata-lock
	     (gethash idx *thread-callns*))
	    t))))


(defun read-thread-cached-object (place-key key)
  (let ((thread-cache
          (bordeaux-threads:with-lock-held (*thread-cache-lock*)
            (gethash (bordeaux-threads:current-thread) *thread-caches*))))
    (when thread-cache
      (let ((place-cache (gethash place-key thread-cache)))
        (when place-cache
          (gethash key place-cache))))))

(defun return-thread-cached-object (place-key key value thread)
  (let* ((thread-cache
           (bordeaux-threads:with-lock-held (*thread-cache-lock*)
             (or (gethash (bordeaux-threads:current-thread) *thread-caches*)
                 (setf (gethash (bordeaux-threads:current-thread)
                                *thread-caches*)
                       (tg:make-weak-hash-table :weakness :key)))))
         (place-cache
           (or (gethash place-key thread-cache)
               (setf (gethash place-key thread-cache)
                     (make-hash-table :test #'equal)))))
    ;; Overwrite it. Keeping the larger, keeping all may be reasonable
    ;; strategies too.
    (setf (gethash key place-cache) value)))

(defmacro with-thread-cached-mat1 ((var tensor &rest args
                                   &key (place :scratch)
                                   (ctype '*default-mat-ctype*)
                                   (displacement 0)
                                   max-size
                                   (initial-element 0) initial-contents)
                                  &body body)
  "Bind VAR to a matrix of DIMENSIONS, CTYPE, etc. Cache this matrix,
  and possibly reuse it later by reshaping it. When BODY exits the
  cached object is updated with the binding of VAR which BODY may
  change.
  There is a separate cache for each thread and each `PLACE` (under
  EQ). Since every cache holds exactly one MAT per CTYPE, nested
  WITH-THREAD-CACHED-MAT often want to use different `PLACE`s. By
  convention, these places are called `:SCRATCH-1`, `:SCRATCH-2`,
  etc."
  (declare (ignore max-size initial-contents))
  (let ((args (copy-list args)))
    (remf args :place)
    (remf args :ctype)
    (remf args :displacement)
    (remf args :initial-element)
    (alexandria:with-unique-names (key)
      (alexandria:once-only (tensor displacement ctype initial-element)
        `(let ((,key (list ,ctype ,initial-element)))
           (with-thread-cached-object1
               (,var ,tensor ,key (make-mat (!shape ,tensor)
                                    :ctype ,ctype
                                    :displacement ,displacement
                                    :initial-element ,initial-element
                                    ,@args)
                :place ,place)
             (setq ,var (adjust! ,var (!shape ,tensor) ,displacement))
	     ; may create new mat
             (locally ,@body)))))))

(defun return-caches ()
  *thread-caches*)

(defun caches-gc ()
  (tg:gc :full t))

(defun free-cache (idx)
  (let* ((caches (bordeaux-threads:with-lock-held (*thread-cache-lock*)
		   (gethash (bordeaux-threads:current-thread) *thread-caches*))))
    (when caches
      (when (gethash idx caches)
	(lock-calln idx)
	(remhash idx caches)))
    nil))

(defmacro with-cache ((var tensor &key (ctype '*default-mat-ctype*)
                      (place :ones))
                     &body body)
  `(with-thread-cached-mat1 (,var ,tensor :place ,place
                                 :ctype ,ctype :initial-element 0.0)
     (let ((,var ,var))
       ,@body)))

;(declaim (inline copy-by-idx))
(defun copy-by-idx (idx mat)
  "When mats are allowed to fail to copy, return mat itself"
  (let ((calln (gethash idx *thread-callns*)))
    (cond
      ((null calln) (copy-mat mat)) ; calln hasn't created yet
      ((null (cachedata-lock calln)) (copy-mat mat)) ; calln's record hasn't finished yet
      ((= 0 (cachedata-calln calln)) mat)
      (T (copy-mat mat)))))

(defmacro with-thread-cached-object1 ((var tensor key initform &key place)
				      &body body)
  (let ((place (or place (gensym (symbol-name 'place)))))
    `(labels ((cached-data (tensor shape? _)
	      (declare (ignore _))
	       (let ((obj (read-thread-cached-object
			   (cl-waffe::waffetensor-idx tensor)
			   (cl-waffe::waffetensor-key tensor))))
		 (if (null obj)
		     (error "cl-waffe.caches:cached-data: The tensor that attempted to read has already cached and cleaned.~%Please remain that calculations must be done in the scope that the tensor has created, including defnode."))
		 (update-calln (cl-waffe::waffetensor-idx tensor))
		 (if shape?
		     (mat-dimensions obj)
		     obj))))
      (let ((,var ,initform))
	 (when (cl-waffe::waffetensor-is-sysconst? ,tensor)
	   (return-thread-cached-object ,place
					,key
					(copy-by-idx
					 ,place
					 (data ,tensor))
					(cl-waffe::waffetensor-thread-data ,tensor))
	   (setf (data ,tensor) #'cached-data)
	   (setf (cl-waffe::waffetensor-key ,tensor) ,key)
	   (setf (cl-waffe::waffetensor-idx ,tensor) ,place))
	 (locally ,@body)))))
