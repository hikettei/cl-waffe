
(in-package :cl-waffe.documents)

(defparameter *tutorials* "")

(with-page *tutorials* "Tutorials"
  (with-section "Tensor"
    (insert "Most deep learning frameworks, represented by PyTorch's Tensor and Chainer's Variables, has their own data structures to store matrices. In cl-waffe, @b(WaffeTensor) is available and defined by Common Lisp's @b(defstruct).")

    (with-section "What can WaffeTensor do?"
      (insert "A")
      (with-evals
	"(setq x (!randn `(3 3)))"
	"(data x)")
      )
    
    (with-section "Parameter and Const"
      (insert "")
      )
    )
  
  (with-section "How does these macros work?, defnode and call."
    (insert "")
    )

  (with-section "Writing Node Extensions"
    (insert "")
    )

  (with-section "MNIST Example"

    ))
