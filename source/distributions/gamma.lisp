
(in-package :cl-waffe)

(defun !gamma (dims k &optional (theta 1.0))
  "Initialize tensor with samples of gamma distribution.

Todo: Use fast algorithms and approximations in response to @cl:param(k).

Example:
@begin[lang=lisp](code)
(!gamma '(10 10) 1.0)
;#Const(((2.155... 3.374... ~ 1.274... 0.147...)        
;                 ...
;        (0.194... 0.081... ~ 0.816... 0.209...)) :mgl t :shape (10 10))
@end[lang=lisp](code)"
  (declare ;(optimize (speed 3))
	   (type cons dims))
  
  ; ↓やる気無くした人 適当な早いアルゴリズム実装してぇ~~
  (const (make-mat dims
		   :initial-contents (numcl:gamma k theta dims))))

