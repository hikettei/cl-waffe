
(in-package :cl-waffe.documents)

#|
  Utils for writing documents
|#

(defmacro with-page (title-binding-symbol
		     title-name
		     &body
		       body)
  `(setq ,title-binding-symbol
	 (with-section ,title-name
	   ,@body)))

(defmacro with-section (title-name
			&body body
			&aux (output-to (gensym)))
  `(with-output-to-string (,output-to)
     (format ,output-to "~%@begin(section)~%@title(~a)" ,title-name)
     (macrolet ((insert (content &rest args)
		  `(format ,',output-to "~%~a" (format nil ,content ,@args)))
		(b (content &rest args)
		  `(format ,',output-to "~%@b(~a)" (format nil ,content ,@args)))
		(image (url)
		  `(format ,',output-to "~%@image[src=\"~a\"]()" ,url))
		(url (url name)
		  `(format ,',output-to "~%@link[uri=\"~a\"](~a)" ,url ,name))
		(item (content)
		  `(format ,',output-to "~%@item(~a)" ,content)))
       (macrolet ((with-section (title-name &body body)
		    `(progn
		       (format ,',output-to "~%@begin(section)~%@title(~a)~%" ,title-name)
		       ,@body
		       (format ,',output-to "~%@end(section)")))
		  (with-enum (&body body)
		    `(progn
		       (format ,',output-to "~%@begin(enum)")
		       ,@body
		       (format ,',output-to "~%@end(enum)")))
		  (with-lisp-code (content)
		    `(format ,',output-to "~%@begin[lang=lisp](code)~%~a~%@end[lang=lisp](code)" ,content))
		  (with-shell-code (content)
		    `(format ,',output-to "~%@begin[lang=shell](code)~%~a~%@end[lang=shell](code)" ,content))

		  )
	 ,@body))
     (format ,output-to "~%@end(section)")))

