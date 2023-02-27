
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

(defun destructive-x-and-y-test (dfunc ndfunc)
  (let* ((x (!randn `(100 100)))
	 (y (!randn `(100 100)))
	 (x-copy (const (mgl-mat:copy-mat (data x))))
	 (y-copy (const (mgl-mat:copy-mat (data y)))))
    (funcall dfunc x y)
    (let ((result (funcall ndfunc x-copy y-copy)))
      (mgl-mat:M= (value x) (value result)))))

(defun broadcasting-destructive-x-and-y-test (dfunc ndfunc)
  (let* ((x (!randn `(100 100 100)))
	 (y (!randn `(1)))
	 (x-copy (const (mgl-mat:copy-mat (data x))))
	 (y-copy (const (mgl-mat:copy-mat (data y)))))
    (funcall dfunc x y)
    (let ((result (funcall ndfunc x-copy y-copy)))
      (mgl-mat:M= (value x) (value result)))))

(defun scale-destructive-x-and-y-test (dfunc ndfunc)
  (let* ((x (!randn `(100 100 100)))
	 (y (const 2.0))
	 (x-copy (const (mgl-mat:copy-mat (data x)))))
    (funcall dfunc x y)
    (let ((result (funcall ndfunc x-copy y)))
      (mgl-mat:M= (value x) (value result)))))

(defun scale-destructive-x-and-y-test1 (dfunc ndfunc)
  (let* ((x (!randn `(100 100 100)))
	 (y (const 2.0))
	 (x-copy (const (mgl-mat:copy-mat (data x)))))
    (funcall dfunc y x)
    (let ((result (funcall ndfunc y x-copy)))
      (mgl-mat:M= (value x) (value result)))))

(defun test-destructive-function (dfunc ndfunc)
  (let* ((x (!randn `(100 100 100)))
	 (x-copy (const (mgl-mat:copy-mat (data x)))))

    (funcall dfunc x)
    (mgl-mat:M= (value x) (funcall ndfunc x-copy))))

(test destructive-scope-test
      (is (!!add-test)))

(test destructive-tests
      (is (destructive-x-and-y-test #'!!add #'!add))
      (is (destructive-x-and-y-test #'!!sub #'!sub))
      (is (destructive-x-and-y-test #'!!mul #'!mul))
      (is (destructive-x-and-y-test #'!!div #'!div)))

(test broadcast-destructive-tests
      (is (broadcasting-destructive-x-and-y-test #'!!add #'!add))
      (is (broadcasting-destructive-x-and-y-test #'!!sub #'!sub))
      (is (broadcasting-destructive-x-and-y-test #'!!mul #'!mul))
      (is (broadcasting-destructive-x-and-y-test #'!!div #'!div)))

(test scale-operation-tests
      (is (scale-destructive-x-and-y-test #'!!add #'!add))
      (is (scale-destructive-x-and-y-test #'!!sub #'!sub))
      (is (scale-destructive-x-and-y-test #'!!mul #'!mul))
      (is (scale-destructive-x-and-y-test #'!!div #'!div)))

(test scale-operation-tests1
      (is (scale-destructive-x-and-y-test1 #'!!add #'!add))
      (is (scale-destructive-x-and-y-test1 #'!!sub #'!sub))
      (is (scale-destructive-x-and-y-test1 #'!!mul #'!mul))
      ;(is (scale-destructive-x-and-y-test1 #'!!div #'!div))
      )


