
(in-package :cl-waffe)


#|
Basic displacement APIs.
|#

#|
+++1+
+++++ =>   1+++++++++
+++++
|#
#|
(defnode ReshapeAndDisplaceTensor (shape displacement)
  :parameters ((displacement displacement :type fixnum)
	       (shape shape :type cons))
  :forward ((x)
	    (reshape-and-displace!
	     (data x)
	     (self shape)
	     (self displacement))
	    (const (data x)))
  :backward ((dy)
	     (list dy)))|#

(defmacro !dotensor (displacement-var tensor)
  ""
  ;memo: displacement=0以外でError/Warning arefを自動でこれにする
  )

(defun !displace ()
  ""

  )
