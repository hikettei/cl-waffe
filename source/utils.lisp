
(in-package :cl-waffe)

(defparameter *gendoc-mode* t
  "Enable this nil for general usage.
Owing to inlined-generic-function's bugs, without this param be t, stackoverflow occurs.")

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
  "Build docstring based on usage and object-type

Todo: Write Doc"
  (with-output-to-string (doc)
    ; Todo ArgsDesc
    (format doc "@begin(section)~%@title(cl-waffe's ~:(~a~): ~a)~%"
	    object-type
	    (waffeobjectusage-name usage))
    (format doc "@b(This structure is an cl-waffe object) ~%@begin(deflist)~%")
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
       (format doc "@term(Forward)~%")
       (format doc "@def(~a)~%"
	       (waffeobjectusage-forward usage))

       (format doc "@term(Call Forward)~%@begin(def)
@begin[code=lisp](code)~%(call (~a) ~a)~%@end[code=lisp](code)~%@end(def)~%"
	       (waffeobjectusage-name usage)
	       (waffeobjectusage-forward-args usage))
       (format doc "@term(Backward description)~%@def(~a)~%"
	       (waffeobjectusage-backward usage))
       (format doc "@term(Call Backward)
@begin(def)~%
@begin[code=lisp](code)~%(call-backward (~a) dy)~%@end[code=lisp](code)~%"
	       (waffeobjectusage-name usage))
       (format doc "Note that: Parameters in node won't be updated.~%@end(def)~%"))
      (:model
       (format doc "@term(Forward)~%")
       (format doc "@def(~a)~%"
	       (waffeobjectusage-forward usage))

       (format doc "@term(Call Forward)~%@begin(def)~%
@begin[code=lisp](code)~%(call (~a) ~a)~%@end[code=lisp](code)~%"
	       (waffeobjectusage-name usage)
	       (waffeobjectusage-forward-args usage))
       (format doc "~%@end(def)~%"))
      (:optimizer
       (format doc "@term(Update)~%")
       (format doc "@def(~a)~%"
	       (waffeobjectusage-update usage))

       (format doc "@term(Call update)~%@begin(def)
use deftrainer and @c((update)), or @c((call-forward (~a) &rest args))~%@end(def)~%"
	       (waffeobjectusage-name usage))
       (format doc "@term(Call zero-grad)~%@begin(def)~%use deftrainer and @c((zero-grad)) or @c((call-backward (~a)))~%@end(def)~%"
	       (waffeobjectusage-name usage)))
      (:trainer
       (format doc "@term(step-model)~%")
       (format doc "@def(~a)~%"
	       (waffeobjectusage-forward usage))
       (format doc "@term(predict)~%")
       (format doc "@def(~a)~%"
	       (waffeobjectusage-predict usage))

       (format doc "@term(Step model)~%@begin(def)
@begin[lang=lisp](code)
(step-model (~a) ~a)
@end[lang=lisp](code)~%@end(def)~%"
	       (waffeobjectusage-name usage)
	       (waffeobjectusage-forward-args usage))
       (format doc "@term(predict)
@begin(def)
@begin[lang=lisp](code)
(predict (~a) &rest args)
@end[lang=lisp](code)
@end(def)~%"
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

cl-waffe automatically generate docstrings.

Todo:Write Docs"
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
