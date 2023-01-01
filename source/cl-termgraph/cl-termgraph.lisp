
(in-package :cl-termgraph)
(use-package :cl-ansi-text)

(deftype line-colors ()
  `(member :black
	   :red
	   :green
	   :yellow
	   :blue
	   :magenta
	   :cyan
	   :white))

(defclass simple-graph-frame ()
  ((width :accessor frame-width
	  :initarg :width
	  :initform nil)
   (height :accessor frame-height
	   :initarg :height
	   :initform nil)))

(defmacro graphline (str)
  `(white ,str :style :background))

(defmethod draw-graph-base ((frame simple-graph-frame))
  (with-slots ((x width) (y height)) frame
    (make-array `(,(1+ x) ,(1+ y)) :initial-element " ")))
