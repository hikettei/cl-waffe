
(in-package :cl-waffe)

(defstruct (WaffeTensor (:constructor
		       tensor
			(value &optional (backend :cpu) &aux (data value) (backend backend) (is-param t)))
		   (:constructor
		       const
			(value &optional (backend :cpu) &aux (data value) (backend backend) (is-const t))))
  data grad backward backend is-param is-const)
