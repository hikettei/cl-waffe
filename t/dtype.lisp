
(in-package :cl-waffe-test)

(defun all-the-operations ()
  (!add (!randn `(10 10)) 1.0)
  (!sub (!randn `(10 10)) 1.0)
  (!mul (!randn `(10 10)) 1.0)
  (!div (!randn `(10 10)) 1.0)
  
  (!add (!randn `(10 10)) (!randn `(10 10))))
