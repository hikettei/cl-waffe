
(in-package :cl-waffe.documents)

(defparameter *overview* "")

(with-page *overview* "Overview"
  (with-section "About cl-waffe"
    (image "https://github.com/hikettei/cl-waffe/blob/main/docs/cl-waffe-logo.png?raw=true")
    (image "https://github.com/hikettei/cl-waffe/actions/workflows/ci.yml/badge.svg")
    (insert "This documentation provides an overview of the development and usage of cl-waffe, based on Common Lisp and mgl-mat.")
    (insert "The primary goal of this project is:")
    
    (with-enum
      (item "Flexible And Efficient Platform in 99% Pure Common Lisp.")
      (item "Make APIs Extensible as possible, enabling users not to depend the standard implementations.")
      (item "Eazy to optimize with Inlined Function."))

    (insert "This framework is designed to be user-friendly first, enabling both begginers and experts in the field of AI to take advantage of capabilities of powerful programming language, Common Lisp.")

    (b "This framework is still under development and experimental. If you are thinking og using it in your products, It would be wiser to use other libraries. True, the author of cl-waffe is not a expert of AI."))


  )
  
