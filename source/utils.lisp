
(in-package :cl-waffe)

; Utils for build documents.
; emmm i guess it is waste of memory usage...?

;(defparameter *object-usages* (make-hash-table))

(defstruct WaffeObjectUsage
  "An structure of restoring waffe objects usage

When you define object (defmodel defnode etc...) all of them will be used for docstrings."
  (name "" :type string)
  (overview "" :type string)
  (args "" :type string)
  (forward "" :type string)
  (forward-args "" :type string)
  (backward "" :type string)
  (update "" :type string)
  (step-model "" :type string)
  (predict "" :type string)
  (next "" :type string)
  (length "" :type string)
  (note "" :type string))
  

(declaim (ftype (function (waffeobjectusage symbol) string) build-docstring))
(defun build-docstring (usage object-type)
  ; in progess
  (with-output-to-string (doc)
    ; Todo ArgsDesc
    (format doc "@begin(section)~%@title(cl-waffe's ~a: ~a)~%"
	    object-type
	    (waffeobjectusage-name usage))
    (format doc "@b(This structure is cl-waffe object) ~%@begin(deflist)~%")
    (format doc "@term(Overview)~% @def(~a)~%"
	    (waffeobjectusage-overview usage))
    
    (unless (equal (waffeobjectusage-note usage) "")
      (format doc "@term(Note)~%@begin(def)~%@u(~a)~%@end(def)~%" (waffeobjectusage-note usage)))

    (format doc "@term(How to Initialize)~% @begin(def)~%@begin[lang=lisp](code)~%(~a ~a) => [~a: ~a]~%@end[lang=lisp](code)~%@end(def)~%"
	    (waffeobjectusage-name usage)
	    (waffeobjectusage-args usage)
	    object-type
	    (waffeobjectusage-name usage))

    (case object-type
      (:node
       (format doc "### Forward~%")
       (format doc "~a~%"
	       (waffeobjectusage-forward usage))

       (format doc "How to Call Forward: `(call (~a) ~a)`~%"
	       (waffeobjectusage-name usage)
	       (waffeobjectusage-forward-args usage))
       (format doc "Backward description: ~a~%"
	       (waffeobjectusage-backward usage))
       (format doc "How to Call Backward: `(call-backward (~a) dy)`~%"
	       (waffeobjectusage-name usage))
       (format doc "Note that: Parameters in node won't be updated.~%"))
      (:model
       (format doc "### Forward~%")
       (format doc "~a~%"
	       (waffeobjectusage-forward usage))

       (format doc "How to Call Forward: `(call (~a) ~a)`~%"
	       (waffeobjectusage-name usage)
	       (waffeobjectusage-forward-args usage))
       (format doc "Note that: Its backward slot will never called.~%"))
      (:optimizer
       (format doc "### Update~%")
       (format doc "~a~%"
	       (waffeobjectusage-update usage))

       (format doc "How to call update: use deftrainer and `(update)`, or `(call-forward (~a) &rest args)`~%"
	       (waffeobjectusage-name usage))
       (format doc "How to call zero-grad: use deftrainer and `(zero-grad)` or `(call-backward (~a))`"
	       (waffeobjectusage-name usage)))
      (:trainer
       (format doc "### step-model~%")
       (format doc "~a~%"
	       (waffeobjectusage-forward usage))
       (format doc "### predict~%")
       (format doc "~a~%"
	       (waffeobjectusage-predict usage))

       (format doc "How to step model: `(step-model (~a) ~a)`~%"
	       (waffeobjectusage-name usage)
	       (waffeobjectusage-forward-args usage))
       (format doc "How to predict: `(predict (~a) &rest args)`~%"
	       (waffeobjectusage-name usage)))
      (:dataset
       (format doc "@term(get-dataset)~%")
       (format doc "@begin(def)~%@begin[lang=lisp](code)~%(get-dataset ~a index) ; => Next Batch~%@end(code)~%@end(def)~%"
	       (waffeobjectusage-name usage))

       (format doc "@term(get-dataset-length)~%")
       (format doc "@begin(def)~%@begin[lang=lisp](code)~%(get-dataset-length ~a) ; => Total length of ~a~%@end(code)~%@end(def)~%"
	       (waffeobjectusage-name usage)
	       (waffeobjectusage-name usage))))
    (format doc "~%@term(Object's slots)@def()~%")
    (format doc "~%@end(deflist)~%@end(section)")))
      
(defmacro with-usage (object-name
		      &key
			(overview "Nothing")
			(args "describe like &rest this")
			(forward "Nothing")
			(step-args "describe like &rest this")
			(backward "Nothing")
			(update "Nothing")
			(step-model "Nothing")
			(predict "Nothing")
			(next "Nothing")
			(length "Nothing")
			(note ""))
  "In :document slot, (for `defmodel, defnode, defoptimizer deftrainer defdataset`) this macro will be useful.

Keyword:
   step-args, arguments for :forward or :step-model, :next.

cl-waffe automatically generate docstrings."
  (make-WaffeObjectUsage
   :name object-name
   :overview overview
   :args args
   :forward forward
   :forward-args step-args
   :backward backward
   :update update
   :step-model step-model
   :predict predict
   :next next
   :note note
   :length length))
  

(defun waffe-help (object)
  "waffe-help explains the usage of object, as long as :document is defined"
  (declare (ignore object)))