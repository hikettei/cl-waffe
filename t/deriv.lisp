
(in-package :cl-waffe-test)

(in-suite :test)

; Tests for backward (its test integrated to optimizers)


(defun aref-backward1 ()
  (let* ((tensor (!randn `(10 10)))
	 (a (parameter (const (mgl-mat:copy-mat (data tensor)))))
	 (b (!aref a 0))
	 (c (!sum b)))
    (backward c)
    (mgl-mat:M=
     (data (!fill `(10) 0.1))
     (data (!aref (const (grad a)) 0)))))

(defun aref-backward2 ()
  (let* ((tensor (!randn `(10 10)))
	 (grad (!mul (!exp tensor) 0.1))
	 (a (parameter tensor))
	 (b (!exp a))
	 (c (!aref b 0))
	 (d (!sum c)))
    (backward d)
    (mgl-mat:M=
     (data (!aref grad 0))
     (data (!aref (const (grad a)) 0)))))

(defun aref-backward3 ()
  (let* ((tensor (!randn `(10 10)))
	 (grad (!mul (!exp tensor) 0.1))
	 (a (parameter tensor))
	 (b (!exp a))
	 (c (!aref b 0))
	 (c (!aref c 0))
	 (c (!aref c 0))
	 (d (!sum c)))
    (backward d)
    (mgl-mat:M=
     (data (!aref grad 0))
     (data (!aref (const (grad a)) 0)))))

(defun aref-backward4 ()
  (let* ((tensor (!randn `(10 10)))
	 (a (parameter tensor))
	 (b (!add a 1.0))
	 (c (!aref b 0))
	 (d (!sum c)))
    (backward d)
    (mgl-mat:M=
     (data (!fill `(10) 0.1))
     (data (!aref (const (grad a)) 0)))))

(defun batchnorm-backward ()
  (let* ((tensor (parameter (!randn `(10 10)))))
    (backward (!sum (call (BatchNorm2D 10) tensor)))
    (grad tensor)))

(test aref-backwards
      (is (aref-backward1))
      (is (aref-backward2))
      (is (aref-backward3))
      (is (aref-backward4)))

(test batchnorm-backward
      (is (batchnorm-backward)))
      
