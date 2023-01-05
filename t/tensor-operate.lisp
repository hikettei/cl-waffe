
(in-package :cl-waffe-test)

(in-suite :test)

(setq a (!randn `(10 10 10 10)))
(setq b (!randn `(10 10 10 10)))


(print (!add a b))
(print (!sub a b))
(print (!mul a b))
(print (!div a b))
(print (!sum a 1))
(print (!mean a 1))
(print (!log a))
(print (!reshape a `(1 100 10 10)))
(print (!pow a 3))
(print (!exp a))
;(print (!dot a b))

(print (!unsqueeze a))
(print a)
(print (!squeeze (!unsqueeze a)))
(print a)
;(print (!matmul a b))
