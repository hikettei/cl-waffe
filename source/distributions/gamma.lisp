
(in-package :cl-waffe)

; when 0<k<1, 
; when k>1, use ziggurat.

(defun !gamma (dims k &optional (theta 1.0))
  "Initializes the new tensor of dims with samples of gamma distribution."
  (declare ;(optimize (speed 3))
	   (type cons dims))
  
  ; ↓やる気無くした人 適当な早いアルゴリズム実装してぇ~~
  (const (make-mat dims
		   :initial-contents (numcl:gamma k theta dims))))

