
(in-package :cl-waffe-test)

(def-suite :caches)
(in-suite :caches)

(defun trace-test-1 ()
  (dotimes (i 10)
    (let ((x (!randn `(10 10))))
      (with-tracing
	(!exp x)
	(!exp x)
	(print (!exp x)))))
  t)

(defun trace-test-2 ()
  (let ((x (!randn `(10 10))))
    (with-tracing
      (let ((y (!exp x)))
	(!exp y)
	(!exp x)))))

(test trace-tests
      (is (trace-test-1))
      (is (trace-test-2))

      )
