
(in-package :cl-waffe.documents)

(defparameter *tutorials* "")

(with-page *tutorials* "Tutorials"
  (with-section "Tensor"
    (insert "Most deep learning frameworks, represented by PyTorch's Tensor and Chainer's Variables, has their own data structures to store matrices. In cl-waffe, @b(WaffeTensor) is available and defined by Common Lisp's @b(defstruct).")

    (with-section "What can WaffeTensor do?"
      (insert "Internally, All matrices created by cl-waffe is a type of mgl-mat, being accessed by the accessor (data tensor).")
      (with-evals
	"(setq x (!randn `(3 3)))"
	"(data x)")

      (insert "In the same way, WaffeTensor can restore scalar object.")
      
      (with-evals
	"(setq x (const 1.0))"
	"(data x)")

      (insert "That is, one of the main roles of WaffeTensor is to be @b(a wrapper for multiple data structures.)")

      (insert "Moreover, WaffeTensor also has these roles:")
      
      (with-enum
	(item "To restore Computation Nodes")
	(item "To restore Gradients")
	(item "To distinguish What Tensor Requires Grads")
	(item "To store Lazy-Evaluated Function."))
      (insert "This is why the definition of WaffeTensor has a large number of slots."))

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
	(insert "I except they're used when...")
	(with-deflist
	  (def "Constant")
	  (term "Datasets, the temporary result of calculations, Parameter which is not necessary to be optimized")

	  (def "Parameter")
	  (term "@b(Trainable Variable)")))))
  
  (with-section "How does these macros work?, defnode and call."
    (insert "The macros @b(defnode) and @b(call) serve as a key component of cl-waffe, since @b(defnode) enables users to define forward and backward propagation in a simple notations and optimize them. If needed, they're inlined via @b(call) macro. Let's get started with this example:")
    (with-eval
      "(defnode ScalarAdd ()
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

    (insert "setting :disassemble-forward or :disassemble-backward t, prints the disassemble of :forward/:backward (only essential parts) respectively. It seems they're enough optimized.")
    
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

    (insert "The defined nodes works as if CLOS object, but it is eazy to inline them.")
    (with-evals
      "(time (call (ScalarAdd) (const 1.0) (const 1.0)))")

    (with-lisp-code
      "Evaluation took:
  0.000 seconds of real time
  0.000005 seconds of total run time (0.000005 user, 0.000000 system)
  100.00% CPU
  11,084 processor cycles
  0 bytes consed")

    (insert "It seems enough inlined and the overhead is enough small considering `ScalarAdd` requires to create computation nodes. This is because call is expanded to:")

    (with-evals
      "(macroexpand `(call (ScalarAdd) (const 1.0) (const 1.0)))")

    (insert "where |call-scalaradd-forward-mgl| is automatically generated function by defnode. For example: If you define the function sadd which calls ScalarAdd Node. You can do:")

    (with-eval
      "(defun sadd (x y)
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
    (insert "We got a large disassembled codes which means: all processes including building computation nodes parts, are correctly inlined. Anyway, the optimization of sadd function is properly working!")

    ; todo: the case when (call node x y)
    ; todo: get-forward-caller get-backward-caller
    )
  
  (with-section "Writing Node Extensions"
    (insert "")
    )

  (with-section "MNIST Example"

    ))
