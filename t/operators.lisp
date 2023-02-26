
(in-package :cl-waffe-test)

(in-suite :test)

; An test codes for destructive/nondestructive operations.


(defun d-test-1 ()
  (let ((x (!ones `(10 10)))
	(y (!ones `(10 10))))
    (!add x y)
    (= 100.0 (data (!sum x)))))

(test non-destructive-test
      (is (d-test-1)))

(defun !!add-test ()
  (let ((x (!ones `(10 10)))
	(y (!ones `(10 10))))
    (!!add x y)
    (= 200.0 (data (!sum x)))))

(test destructive-test
      (is (!!add-test)))
