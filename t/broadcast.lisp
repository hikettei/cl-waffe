
(in-package :cl-waffe-test)

(in-suite :test)

(setq a (!randn `(100 100)))
(setq b (!randn `(100 1)))

(!add a b)
