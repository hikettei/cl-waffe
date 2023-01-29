
(defpackage :cl-waffe.caches
  (:use :cl :mgl-mat)
  (:export
   #:with-cache
   #:return-caches
   #:free-cache
   #:caches-gc))

(in-package :cl-waffe.caches)

; todo depends on tg, bordeaux-threads
(defvar *thread-caches* (tg:make-weak-hash-table :weakness :key))
(defvar *thread-cache-lock* (bordeaux-threads:make-lock "thread cache lock"))

(defun borrow-thread-cached-object (place-key key)
  (let ((thread-cache
          (bordeaux-threads:with-lock-held (*thread-cache-lock*)
            (gethash (bordeaux-threads:current-thread) *thread-caches*))))
    (when thread-cache
      (let ((place-cache (gethash place-key thread-cache)))
        (when place-cache
          (prog1 (gethash key place-cache)
	    (remhash key place-cache)))))))

(defun return-thread-cached-object (place-key key value)
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

(defmacro with-thread-cached-mat1 ((var dimensions &rest args
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
      (alexandria:once-only (dimensions displacement ctype initial-element)
        `(let ((,key (list ,ctype ,initial-element)))
           (with-thread-cached-object1
               (,var ,key (make-mat ,dimensions
                                    :ctype ,ctype
                                    :displacement ,displacement
                                    :initial-element ,initial-element
                                    ,@args)
                :place ,place)
             (setq ,var (adjust! ,var ,dimensions ,displacement))
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
      (let ((caches-for-idx (gethash idx caches)))
	(when caches-for-idx
	  (remhash idx caches))
	))
    nil))

(defmacro with-cache ((var dimensions &key (ctype '*default-mat-ctype*)
                      (place :ones))
                     &body body)
  `(with-thread-cached-mat1 (,var ,dimensions :place ,place
                                 :ctype ,ctype :initial-element 0.0)
     (let ((,var ,var))
       ,@body)))

(defmacro with-thread-cached-object1 ((var key initform &key place) &body body)
  (let ((place (or place (gensym (symbol-name 'place)))))
    (progn
      `(let ((,var (or (borrow-thread-cached-object ,place ,key)
                       ,initform)))
         (unwind-protect
              (locally ,@body)
           (return-thread-cached-object ,place ,key ,var))))))
