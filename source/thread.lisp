
(in-package :cl-waffe)

(defparameter *lparallel-kernel* nil)

(defmacro set-lparallel-kernel (num-cores)
  `(setf *lparallel-kernel* (lparallel:make-kernel ,num-cores)))

