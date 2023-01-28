

(defpackage :cl-waffe.caches
  (:use :cl :mgl-mat)
  (:export
   #:with-cache))

(in-package :cl-waffe.caches)

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
             (locally ,@body)))))))

(defmacro with-cache ((var dimensions &key (ctype '*default-mat-ctype*)
                      (place :ones))
                     &body body)
  `(with-thread-cached-mat1 (,var ,dimensions :place ,place
                                 :ctype ,ctype :initial-element 0.0)
     (let ((,var ,var))
       ,@body)))

(defmacro with-thread-cached-object1 ((var key initform &key place) &body body)
  (let ((place (or place (gensym (symbol-name 'place)))))
    (alexandria:once-only (key)
      `(let ((,var (or (mgl-mat::borrow-thread-cached-object ,place ,key)
                       ,initform)))
         (unwind-protect
              (locally ,@body)
           (mgl-mat::return-thread-cached-object ,place ,key ,var))))))
