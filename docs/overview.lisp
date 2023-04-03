
(in-package :cl-waffe.documents)

(with-page *overview* "Overview"
  (with-section "About This Project"
    (image "https://github.com/hikettei/cl-waffe/blob/main/docs/cl-waffe-logo.png?raw=true")
    (image "https://github.com/hikettei/cl-waffe/actions/workflows/ci.yml/badge.svg")
    (insert "This documentation provides an overview of the development and usage of cl-waffe, based on Common Lisp and mgl-mat.")
    (insert "The primary goal of this project is:")
    
    (with-enum
      (item "Flexible And Efficient Platform in 99% Pure Common Lisp.")
      (item "Make APIs Extensible as possible, enabling users not to depend the standard implementations.")
      (item "Eazy to optimize with Inlined Function."))

    (insert "This framework is designed to be user-friendly first, enabling both beginners and experts in the field of AI to take advantage of capabilities of powerful programming language, Common Lisp.")

    (b "This framework is still under development and experimental. If you are thinking on using it in your products, It would be wiser to use other libraries. True, the author of cl-waffe is not a expert of AI."))

  (with-section "Links"
    (url "https://github.com/hikettei/cl-waffe" "Official Github Repository")
    (url "https://hikettei.github.io/cl-waffe-docs/docs/overview.html" "The Documentation")
    (url "https://github.com/hikettei/cl-waffe/tree/main/tutorials/jp" "Tutorial Notebooks (Written in Japanese)")
    (url "https://github.com/hikettei/cl-waffe/benchmark/Result.md" "Benchmarks"))

  (with-section "Workloads"
    (with-enum
      (item "Full Optimization")
      (item "save models with npz")
      (item "🎉 release cl-waffe v0.1")
      (item "more standard impls")
      ))
  
  (with-section "Sections"
    (insert "The list of sections")
    (with-deflist
      (def "Tutorials")
      (term "a")))

  (with-section "LLA Backend"
    (insert "cl-waffe's matrix operations are performed via mgl-mat, and mgl-mat uses LLA. Accordingly, cl-waffe's performance hinges on mgl-mat and LLA's performance.")
    
    (insert "The most recommended one is OpenBLAS. Append following in your setup files (e.g.: ~~/.roswell/init.lisp, ~~/.sbclrc). For more details, visit the official repositories.")

    (url "https://github.com/tpapp/lla" "LLA")
    (url "https://github.com/melisgl/mgl-mat" "mgl-mat")

    (with-lisp-code
      "(defvar *lla-configuration* '(:libraries (\"/usr/local/opt/openblas/lib/libblas.dylib\")))"))

  (with-section "When Memory Heap Is Exhasted?"
    (insert "The additional setting of dynamic-space-size would be required since training deep learning models consumes a lot of space.")
    (insert "For Example, Roswell and SLIME respectively.")
    (with-shell-code
      "$ ros config set dynamic-space-size 4gb")
    (with-lisp-code
      "(setq slime-lisp-implementations '((\"sbcl\" (\"sbcl\" \"--dynamic-space-size\" \"4096\"))))")
    (insert "should work. However, Improving memory usage is one of my concerns.")))

