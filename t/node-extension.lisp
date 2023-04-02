
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

#|
Note: The macro call is static, which means: backends that `call` can use is determined when `call` is expanded.

So If you added an external backend, you need to redefine function exported by cl-waffe (esp, fully inlined one e.g.: !add).
|#
(defun operate-in-extension ()
  (with-backend :my-extension
    (if (= (data (call (cl-waffe::AddTensor) (const 1) (const 1))) 200)
	t
	nil)))

(defun operate-restart-test () ; if the operation doesn't exists...
  (with-backend :does-not-exists
    (= (data (call (cl-waffe::AddTensor) (const 1) (const 1))) 2)))

#|
(defun operate-restart-test1 ()
  (handler-case (let ((*restart-non-exist-backend* nil))
		  (with-backend :does-not-exists
		    (!add 1 1)
		    nil))
    (Backend-Doesnt-Exists (c)
      (print c)      
      t)))
|#

(test extension-tests
      (is (operate-in-mgl))
      (is (operate-in-extension))
      (is (operate-restart-test))
      ;(is (operate-restart-test1))
      )

