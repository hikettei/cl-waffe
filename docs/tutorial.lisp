
(in-package :cl-waffe.documents)

#|
 To Add: The reference of docstrings.
|#
(with-page *tutorials* "Tutorials"
  (with-section "Introducing WaffeTensor"
    (insert "Most deep learning frameworks, represented by PyTorch's Tensor and Chainer's Variables, has their own data structures to store matrices. In cl-waffe, @b(WaffeTensor) is available and defined by Common Lisp's @b(defstruct).")

    (insert "~%~%⚠️ There is no guarantee that this design is technically mature.")

    (with-section "What can WaffeTensor do?"
      (insert "Internally, All matrices created by cl-waffe is a type of mgl-mat, being accessed by the accessor (data tensor).")
      (with-evals
	"(setq x (!randn `(3 3))) ; WaffeTensor"
	"(data x) ;mgl-mat:mat")

      (insert "In the same way, WaffeTensor can restore scalar object.")
      
      (with-evals
	"(setq x (const 1.0)) : WaffeTensor"
	"(data x) ; single-float")

      (insert "That is, one of the main roles of WaffeTensor is to be @b(a wrapper for multiple data structures.)")

      (insert "~%~%You may well feel it is just rebundant for waffetensor to be only a wrapper. Of course, WaffeTensor has also these roles:")

      (with-section "To Restore Computation Nodes"
	(insert "Operations performed via cl-waffe, creates a @b(comutation nodes). This can all be extended by the defnode and call macros described the defnode and call section.")
	(with-eval
	  "
(let ((a (const 1.0))
      (b (const 1.0)))
  (!add a b))")
	(insert "When gradient is not required (e.g.: predict), the macro @c((with-no-grad)) would be useful.")
	(with-eval
	  "
(with-no-grad
    (let ((a (const 1.0))
	  (b (const 1.0)))
      (!add a b)))"))

      (with-section "To Restore Gradients"
	(insert "WaffeTensors which created by @c((parameter tensor)) macro, posses the gradients, where you can get via `(backward out)`")

	(with-evals
	  "(setq a (parameter (!randn `(3 3))))"
	  "(setq b (parameter (!randn `(3 3))))"
	  "(setq c (parameter (!randn `(3 3))))"
	  "(setq z (!sum (!add (!mul a b) c))) ; computes z=a*b + c, and summarize it."
	  "(backward z)"
	  "(grad a)"
	  "(grad b)"
	  "(grad c)")

	(insert "(backward out) called inside of (with-verbose &body body) macro, will display how the computation nodes are traced. It would be helpful for debugging."))

      (with-section "To distinguish What Tensor Requires Gradients"
	(insert "WaffeTensor that requires gradients, are represented by @c((parameter tensor)), on the other hand, don't requires one are @c((const)). Then, Computational nodes that have no parameters at the destination of back propagation do not need to keep a copy for gradient creation during forward propagation or to perform back propagation in the first place. WaffeTensor determines this dynamically during forward propagation."))

      (with-section "To Store Lazy-Evaluated Object"
	(insert "You may notice that: some operators, like !transpose, creates lazy-evaluated tensor when get started with cl-waffe.")
	(with-evals
	  "(!transpose (!randn `(3 1)))")

	(insert "They behaves as if they're normal tensor (In fact, !shape !dims etc... works as usual), but aren't evaluated until (value tensor) is called.")

	(with-evals
	  "(setq transpose (!transpose (!randn `(3 1))))"
	  "(value transpose)"
	  "transpose")

	(insert "This property helps to reduce the cost of !transpose before !matmul")))

    (with-section "Parameter and Const"
      (insert "There are two types of WaffeTensor, parameter and constant. The parameter creates gradient when (backward out) is called, on the other hand, the constant doesn't.")
      (with-section "Initialize Constants"
	(insert "cl-waffe provides various ways to initialize constants. For example, `!randn` initializes the new tensor of the given dims with sampling the standard distribution, where var=0.0, stdev=1.0. !beta samples the beta distribution with the given alpha and beta.")
	(with-evals
	  "(!randn `(10 10))"
	  "(!beta `(10 10) 2.0 1.0)")
	(insert "WaffeTensors we obtain from standard initializing methods are Constant. In general, cl-waffe provides the constructor (const value). The given value is coerced to properly types. In this example, we obtain mgl-mat from simple-array.")
	(with-evals
	  "(const (make-array `(3 3)))"))

      (with-section "Initialize Parameter"
	(insert "Parameters are initialized via the macro (parameter tensor), which makes the given tensor parameter.")
	(with-evals
	  "(parameter (!randn `(10 10)))"))

      (with-section "Parameter vs Constant"
	(insert "Excepted Usage of them is:")
	(with-deflist
	  (def "Constant")
	  (term "Datasets, the temporary result of calculations, Parameter which is not necessary to be optimized.")

	  (def "Parameter")
	  (term "Trainable Variables, to be optimized by @b(optimizers) defined by defoptimizer.")))))
  
  (with-section "defnode and call"
    (insert "The macros @b(defnode) and @b(call) serve as a key component of cl-waffe, since @b(defnode) enables users to define forward and backward propagation in a simple notations and optimize them. If needed, they're inlined via @b(call) macro. Let's get started with this example. it defines a computation node that finds the sum of two single-float values.")
    (with-eval
      "
(defnode ScalarAdd ()
  :disassemble-forward t
  :forward-declaim (declaim (ftype (function (ScalarAdd waffetensor waffetensor) waffetensor) :forward))
  :forward ((x y)
	    (let ((x (data x))
		  (y (data y)))
	      (declare (type single-float x y))
	      (const (+ x y))))
  :disassemble-backward t
  :backward-declaim (declaim (type (function (ScalarAdd waffetensor) list) :backward))
  :backward ((dy) (list dy dy)))")

    (insert "Through this macro, these structures and functions are defined:")

    (with-enum
      (item "The structure, ScalarAdd")
      (item "The constructor function, (ScalarAdd)")
      (item "The function, (call-scalaradd-forward-mgl self x y) where self is a strucure ScalarAdd")
      (item "The function, (call-scalaradd-backward-mgl self dy) where self is a structure ScalarAdd."))

    (insert "Setting :disassemble-forward or :disassemble-backward t, prints the disassemble of :forward/:backward (only essential parts) respectively. From the result below, it seems to be optimized enough...")
    
    (with-lisp-code
      "; disassembly for #:|nodedebug9718|
; Size: 148 bytes. Origin: #x540A110F                         ; #:|nodedebug9718|
; 0F:       498B4510         MOV RAX, [R13+16]                ; thread.binding-stack-pointer
; 13:       488945F8         MOV [RBP-8], RAX
; 17:       4883EC10         SUB RSP, 16
; 1B:       488B55F0         MOV RDX, [RBP-16]
; 1F:       B902000000       MOV ECX, 2
; 24:       48892C24         MOV [RSP], RBP
; 28:       488BEC           MOV RBP, RSP
; 2B:       B802AC3650       MOV EAX, #x5036AC02              ; #<FDEFN DATA>
; 30:       FFD0             CALL RAX
; 32:       480F42E3         CMOVB RSP, RBX
; 36:       4C8BC2           MOV R8, RDX
; 39:       4C8945E0         MOV [RBP-32], R8
; 3D:       4883EC10         SUB RSP, 16
; 41:       488B55E8         MOV RDX, [RBP-24]
; 45:       B902000000       MOV ECX, 2
; 4A:       48892C24         MOV [RSP], RBP
; 4E:       488BEC           MOV RBP, RSP
; 51:       B802AC3650       MOV EAX, #x5036AC02              ; #<FDEFN DATA>
; 56:       FFD0             CALL RAX
; 58:       480F42E3         CMOVB RSP, RBX
; 5C:       4C8B45E0         MOV R8, [RBP-32]
; 60:       4180F819         CMP R8B, 25
; 64:       7538             JNE L1
; 66:       66490F6ED0       MOVQ XMM2, R8
; 6B:       0FC6D2FD         SHUFPS XMM2, XMM2, #4r3331
; 6F:       80FA19           CMP DL, 25
; 72:       7403             JEQ L0
; 74:       CC51             INT3 81                          ; OBJECT-NOT-SINGLE-FLOAT-ERROR
; 76:       08               BYTE #X08                        ; RDX(d)
; 77: L0:   66480F6ECA       MOVQ XMM1, RDX
; 7C:       0FC6C9FD         SHUFPS XMM1, XMM1, #4r3331
; 80:       F30F58CA         ADDSS XMM1, XMM2
; 84:       660F7ECA         MOVD EDX, XMM1
; 88:       48C1E220         SHL RDX, 32
; 8C:       80CA19           OR DL, 25
; 8F:       B902000000       MOV ECX, 2
; 94:       FF7508           PUSH QWORD PTR [RBP+8]
; 97:       B802DD3650       MOV EAX, #x5036DD02              ; #<FDEFN CONST>
; 9C:       FFE0             JMP RAX
; 9E: L1:   CC51             INT3 81                          ; OBJECT-NOT-SINGLE-FLOAT-ERROR
; A0:       20               BYTE #X20                        ; R8(d)
; A1:       CC10             INT3 16                          ; Invalid argument count trap")

    (with-lisp-code
      "; disassembly for #:|nodedebug9739|
; Size: 84 bytes. Origin: #x541BA04C                          ; #:|nodedebug9739|
; 4C:       498B4510         MOV RAX, [R13+16]                ; thread.binding-stack-pointer
; 50:       488945F8         MOV [RBP-8], RAX
; 54:       4D896D28         MOV [R13+40], R13                ; thread.pseudo-atomic-bits
; 58:       498B5558         MOV RDX, [R13+88]                ; thread.cons-tlab
; 5C:       488D4220         LEA RAX, [RDX+32]
; 60:       493B4560         CMP RAX, [R13+96]
; 64:       772E             JNBE L2
; 66:       49894558         MOV [R13+88], RAX                ; thread.cons-tlab
; 6A: L0:   48893A           MOV [RDX], RDI
; 6D:       48897A10         MOV [RDX+16], RDI
; 71:       48C7421817010050 MOV QWORD PTR [RDX+24], #x50000117  ; NIL
; 79:       488D4217         LEA RAX, [RDX+23]
; 7D:       48894208         MOV [RDX+8], RAX
; 81:       80CA07           OR DL, 7
; 84:       4D316D28         XOR [R13+40], R13                ; thread.pseudo-atomic-bits
; 88:       7402             JEQ L1
; 8A:       CC09             INT3 9                           ; pending interrupt trap
; 8C: L1:   488BE5           MOV RSP, RBP
; 8F:       F8               CLC
; 90:       5D               POP RBP
; 91:       C3               RET
; 92:       CC10             INT3 16                          ; Invalid argument count trap
; 94: L2:   6A20             PUSH 32
; 96:       FF142528050050   CALL [#x50000528]                ; #x52A005B0: LIST-ALLOC-TRAMP
; 9D:       5A               POP RDX
; 9E:       EBCA             JMP L0")

    (insert "Nodes defined this macro, works as if CLOS class, and they can have :parameters. However, what makes defnode distinct from them is that:")
    (with-evals
      "(time (call (ScalarAdd) (const 1.0) (const 1.0)))"
      "(time (+ 1.0 1.0))")

    (with-lisp-code
      "Evaluation took:
  0.000 seconds of real time
  0.000005 seconds of total run time (0.000005 user, 0.000000 system)
  100.00% CPU
  11,084 processor cycles
  0 bytes consed")

    (with-lisp-code
	"Evaluation took:
  0.000 seconds of real time
  0.000001 seconds of total run time (0.000000 user, 0.000001 system)
  100.00% CPU
  422 processor cycles
  0 bytes consed")

    (insert "Nodes called by the macro @c((call)) are fully inlined, (like CL's inline-generic-function, static-dispatch). Considering ScalarAdd builds computation node in addition to summing up the arguments, these overheads are enough small. Here's how I achieve this behaviour:")
    
    (with-evals
      "(macroexpand `(call (ScalarAdd) (const 1.0) (const 1.0)))")

    (insert "The function call-forward-scalaradd-mgl seems to be inlined. This is because @c((call)) can detect the type of node in the compile time. This leads one of the key propeties, @b(easy to optimise). The functions via defnode and call are optimized like:")

    (with-eval
      "
(defun sadd (x y)
    (declare (optimize (speed 3) (safety 0))
             (type single-float x y))
        (call (ScalarAdd) (const x) (const y)))")

    (with-lisp-code
      "(disassemble #'sadd)

; disassembly for SADD
; Size: 943 bytes. Origin: #x541AFCAE                         ; SADD
; AFCAE:       488975F0         MOV [RBP-16], RSI
; AFCB2:       4883EC10         SUB RSP, 16
.
.
(Omitted)")
    
    (insert "We got a large disassembled codes which means: all processes including building computation nodes parts, are correctly inlined. Anyway, the optimization of sadd function is properly working!. Note that the case when the type of given nodes aren't determined in compile time, call behaviours the different from this.")

    (with-eval
      "
(let ((node (ScalarAdd)))
    (macroexpand `(call node (const 1.0) (const 1.0))))")

    (insert "The expanded equation was slightly more complicated. Anyway, the most important part is @c((APPLY #'CALL-INLINED-FORWARD MODEL INPUTS)). In short, call-inlined-forward is like:")

    (with-lisp-code
      "(defun call-inlined-forwrd (model &rest inputs)
    (typecase model
        (addtensor (call-addtensor-forward-mgl ...))
        (scalaradd (call-scalaradd-forward-mgl ...))
        (T ; ... If this is first trying, Redefine call-inline-forward and try again
        )))")

    (insert "It may be misleading but simultaneously the most simple example. Of course they're inlined. And call-inlined-forward are automatically redefined when:")

    (with-enum
      (item "The new backend is defined.")
      (item "The node you specified doesn't match any nodes."))

    (insert "That is, No need to pay attention to when they are inlined.")

    (with-eval
      "(let ((node (ScalarAdd)))
    (time (call node (const 1.0) (const 1.0))))")

    (with-lisp-code
      "Evaluation took:
  0.000 seconds of real time
  0.000005 seconds of total run time (0.000005 user, 0.000000 system)
  100.00% CPU
  10,502 processor cycles
  0 bytes consed")

    (insert "It works the same as the first example, the overhead is enough small.")
    (insert "(P.S.: I was told that it is impossible for SBCL to optimize a CASE of several thousand lines. The assumption is that the more nodes defined in cl-waffe, the less performance we got. In my own benchmarks, I felt it was doing well enough on the second call, but if it is slow, I know how to make it faster.)")
    
    (insert "~%~%By the way, defnode's forward slot can require &rest arguments. However, @c((call)) is a macro, so that we can't use apply. Is there no way to call it with &rest arguments? No, @c(get-forward-caller) and @c(get-backward-caller) is available to get the function object itself. In cl-waffe's implementation, !concatenate requires an &rest arguments.")

    (insert "TO ADD: The link to get-forward-caller...")

    (with-lisp-code "
(defun !concatenate (axis &rest tensors)
  (declare (optimize (speed 3))
	   (type fixnum axis))
  (let* ((node (ConcatenateTensorNode axis))
	 (caller (get-forward-caller node)))
    (apply caller node tensors)))"))
  
  (with-section "Writing Node Extensions"
    (insert "You may notice that the functions generated by defnode has the suffix, mgl. This indicates the backend cl-waffe uses. (mgl = mgl-mat).")
    (insert "~%~%If the existing implementation of nodes aren't suitable for your usage, replace them. and cl-waffe provides the ecosystem to manage these additional implementation, I call it backend. For example, you can replace my broadcasting implementation with another fast implementation method. Let's create a double-float version of AddScalar.")

    (with-eval
      "
(define-node-extension ScalarAdd
	     :backend :double-float
	     :forward-declaim (declaim (ftype (function (ScalarAdd waffetensor waffetensor) waffetensor) :forward))
	     :forward ((x y)
	    (let ((x (data x))
		  (y (data y)))
	      (declare (type double-float x y))
	      (const (+ x y))))
	     :backward-declaim (declaim (type (function (ScalarAdd waffetensor) list) :backward))
	     :backward ((dy) (list dy dy)))")

    (insert "And receive this:")

    (with-lisp-code
      "[INFO] Inlining call-forward... Total Features: 64
To disable this, set cl-waffe:*ignore-inlining-info* t

[INFO] Inlining call-backward... Total Features: 64
To disable this, set cl-waffe:*ignore-inlining-info* t")

    (insert "It's all done. The backends you defined can be switched via (with-backend backend-name &body body) macro. Let's check how call expands it.")

    (with-evals
      "
(with-backend :double-float
    (macroexpand `(call (ScalarAdd) (const 1.0d0) (const 1.0d0))))")

    (insert "There's an additional case generated, depending on *default-backend*.")

    (with-evals
      "
(with-backend :double-float
    (time (call (scalarAdd) (const 1.0d0) (const 1.0d0))))
")
    (with-lisp-code
      "Evaluation took:
  0.000 seconds of real time
  0.000005 seconds of total run time (0.000005 user, 0.000000 system)
  100.00% CPU
  9,814 processor cycles
  0 bytes consed")

    (insert "Adding new backends is no pain for cl-waffe!"))

  (with-section "MNIST Example"
    (insert "Using features that I introduced, we can training MNIST. In practive, we need more features to implement it more simply: defmacro")

    (with-section "define your model"
      (with-evals "
(defmodel MLP (activation)
  :parameters ((layer1   (cl-waffe.nn:denselayer (* 28 28) 512 T activation))
	       (layer2   (cl-waffe.nn:denselayer 512 256 T activation))
	       (layer3   (cl-waffe.nn:linearlayer 256 10 T)))
  :forward ((x)
	    (with-calling-layers x
	      (layer1 x)
 	      (layer2 x)
	      (layer3 x))))"
      "(MLP :relu)"))

    (with-section "define your trainer"
      (with-evals "
(deftrainer MLPTrainer (activation lr)
  :model          (MLP activation)
  :optimizer      cl-waffe.optimizers:Adam
  :optimizer-args (:lr lr)
  :step-model ((x y)
	       (zero-grad)
	       (let ((out (cl-waffe.nn:softmax-cross-entropy (call (model) x) y)))
		 (backward out)
		 (update)
		 out))
 :predict ((x)(call (model) x)))"
	"(setq trainer (MLPTrainer :relu 1e-3))"
	"(slot-value trainer 'cl-waffe::optimizer)"))

    
    ))
