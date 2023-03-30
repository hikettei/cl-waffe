
(in-package :cl-waffe-test)

(defun all-the-operations (&aux
			     (target-tensor1 (!zeros `(10 10)))
			     (target-tensor (!ones  `(10 10))))
  (macrolet ((test-on-onearg-fn (fname)
	       `(,fname target-tensor))
	     (test-on-onearg-fn1 (fname)
	       `(,fname target-tensor1)))

    (!add (!randn `(10 10)) 1.0)
    (!sub (!randn `(10 10)) 1.0)
    (!mul (!randn `(10 10)) 1.0)
    (!div (!randn `(10 10)) 1.0)
    
    (!add (!randn `(10 10)) (!randn `(10 10)))
    (!sub (!randn `(10 10)) (!randn `(10 10)))
    (!mul (!randn `(10 10)) (!randn `(10 10)))
    (!div (!randn `(10 10)) (!ones `(10 10)))

    (!add (!randn `(10 1)) (!randn `(10 10)))
    (!sub (!randn `(10 1)) (!randn `(10 10)))
    (!mul (!randn `(10 1)) (!randn `(10 10)))
    (!div (!randn `(10 1)) (!ones `(10 10)))
    
    (test-on-onearg-fn !sin)
    (test-on-onearg-fn !cos)
    (test-on-onearg-fn !tan)
    
    (test-on-onearg-fn !asin)
    (test-on-onearg-fn !acos)
    (test-on-onearg-fn !atan)
    
    (test-on-onearg-fn !sinh)
    (test-on-onearg-fn !cosh)
    (test-on-onearg-fn !tanh)

    (test-on-onearg-fn !asinh)
    (test-on-onearg-fn !acosh)
    (test-on-onearg-fn1 !atanh)

    (test-on-onearg-fn !log)
    (test-on-onearg-fn !exp)

    (test-on-onearg-fn !relu)
    (test-on-onearg-fn !leakey-relu)
    (test-on-onearg-fn !gelu)
    (test-on-onearg-fn !swish)

    (!aref target-tensor '(0 1))
    (setf (!aref target-tensor '(0 1)) (!ones `(10)))

    (!sum target-tensor)
    (!mean target-tensor 1 t)

    (!pow target-tensor 2)
    (value (!transpose target-tensor))
    (!transpose1 target-tensor)

    (!reshape target-tensor `(1 100))
    (!concatenate 1 target-tensor target-tensor1)
    (!split target-tensor 1)

    (!softmax target-tensor)

    (!matmul (!randn `(10 10 10)) (!randn `(10 10 10)))

    (!argmax target-tensor)
    (!argmin target-tensor)

    (!abs target-tensor)
    (!sqrt target-tensor)
    
    
    
    
    ))


(all-the-operations)
