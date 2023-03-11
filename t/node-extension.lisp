
(in-package :cl-waffe-test)

(in-suite :test)

(define-node-extension cl-waffe::AddTensor
  :backend :my-extension
  :forward ((x y)
            (const (+ 100 100)))
  :backward ((dy)
             (list dy dy)))

(defun operate-in-mgl ()
  (with-backend :mgl
    (= (data (!add 1 1)) 2)))

(defun operate-in-extension ()
  (with-backend :my-extension
    (= (data (!add 1 1)) 200)))

(defun operate-restart-test () ; if the operation doesn't exists...
  (with-backend :does-not-exists
    (= (data (!add 1 1)) 2)))

(test extension-tests
      (is (operate-in-mgl))
      (is (operate-in-extension))
      (is (operate-restart-test)))

