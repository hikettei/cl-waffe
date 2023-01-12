
(in-package :cl-waffe-test)

(in-suite :test)

(setf *ignore-optimizer* t)

(setq a (!randn `(1024 1024)))
(setq b (!randn `(1024 1024)))

(setq c (!randn `(1024 1024 5)))

(setq s (!random `(1024 1024) `(1.0 5.0)))

(setq scalar 3)

(print "!add")
(time (!add a b))
(print "original")
(time (mgl-mat:axpy! 1.0 (data a) (data b)))

(print "scalar-add")
(time (!add scalar a))

;(print "multi-dimension-add")
;(time (!add a c))

(print "!sub")
(time (!sub a b))
(print "scalar-sub")
(time (!sub scalar b))

(print "!mul")
(time (!mul a b))
(print "scalar-mul")
(time (!mul scalar b))

(print "!div")
(time (!div a b))
(print "scalar-div")
(time (!div a scalar))

(print "!log")
(time (!log s))

(print "!sqrt")
(time (!sqrt s))
;(print (!reshape a `(1 100 10 10)))
;(print (!pow a 3))
;(print (!exp a))
;(print (!dot a b))

;(print (!unsqueeze a))
;(print a)
;(print (!squeeze (!unsqueeze a)))
;(print a)
;(print (!matmul a b))
