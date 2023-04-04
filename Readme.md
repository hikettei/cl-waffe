[![CI](https://github.com/hikettei/cl-waffe/actions/workflows/ci.yml/badge.svg)](https://github.com/hikettei/cl-waffe/actions/workflows/ci.yml)

<p align="center">
    <a href="https://github.com/hikettei/cl-waffe">
        <img alt="Logo" src="https://hikettei.github.io/cl-waffe-docs/cl-waffe.png" width="45%">
    </a>
    <br>
    <h3 align="center">Deep Learning Framework for Common Lisp</h3>
    <p align="center">
    <a href="https://hikettei.github.io/cl-waffe-docs/docs/overview.html"><strong>Documentations »</strong></a>
    <br />
    <br />
    <a href="https://github.com/hikettei/cl-waffe/issues">Issues</a>
    ·
    <a href="https://github.com/hikettei/cl-waffe/blob/main/benchmark/Result.md">Benchmarks</a>
    ·
    <a href="https://github.com/hikettei/cl-waffe/tree/main/tutorials/jp">Tutorials(JP)</a>
  </p>
</p>


# About This Project

cl-waffe is a deep learning framework with modern APIs for Common Lisp, based on [mgl-mat](https://github.com/melisgl/mgl-mat). 

This framework is 100% written in Common Lisp (ignored BLAS/CUBLAS parts). As a result, it is extremely easy to extend the features as needed. However, 
The framework currently has a limited set of features, and I am working to expand its capabilities in future releases.

**⚠️ This framework is still under development and experimental. If you are thinking on using it in your products, It would be wiser to use other libraries.** It should be noted that the author of cl-waffe is not an AI expert.  Also, not having GPUs, I can't test my framework on cuda ><. CUDA support is a little further along. (Ignoring some operations like Embedding, most operations are performed via [mgl-mat](https://github.com/melisgl/mgl-mat), so it should work without any modifications.)

# TOC

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [MNIST Example](#mnist-example)
- [Features](#features)
  - [Broadcasting](#broadcasting)
  - [Destructive APIs with a Simple Rule.](#destructive-apis-with-a-simple-rule)
  - [Useful APIs like Numpy/PyTorch.](#useful-apis-like-numpypytorch)
  - [Automatic Differentiation](#automatic-differentiation)
  - [Useful Lazy-Evaluation System](#useful-lazy-evaluation-system)
  - [Eazy to optimize](#eazy-to-optimize)
  - [Extensible APIs](#extensible-apis)
  - [Switchable Backends](#switchable-backends)
- [Install](#install)
    - [Install via Github](#install-via-github)
    - [Install via Roswell](#install-via-roswell)
    - [Install via Ultralisp](#install-via-ultralisp)
- [Contributing](#contributing)
    - [Prerequisites](#prerequisites)
    - [Running the tests](#running-the-tests)
  - [Lakefile](#lakefile)
- [Run MNIST With Roswell](#run-mnist-with-roswell)
- [Currently Problems/Todo](#currently-problemstodo)
- [Goals](#goals)
- [Acknowledgements](#acknowledgements)
- [Author](#author)
- [Environment](#environment)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


# What cl-waffe does?

cl-waffe is the two sides of the coin:
1. As a **matrix operation library**.
2. As a **deep learning framework**.

## cl-waffe as a matrix operation library

Speaking of the former, cl-waffe aimed to wrap the existing Common Lisp matrix operation libraries with Numpy/PyTorch like APIs. And reduce overheads between cl-waffe and these libraries.

So, If you are considering contributing to cl-waffe in terms of boosting its performance, the first thing you should do is **to contribute libraries cl-waffe uses**, especially, [mgl-mat](https://github.com/melisgl/mgl-mat).

As I mentioned at [this reddit post](https://www.reddit.com/r/Common_Lisp/comments/124da1l/does_anyone_have_any_interest_in_my_deeplearning/?utm_source=share&utm_medium=web2x&context=3), `a solid foundation must first be in place` to accomplish deep learning on Common Lisp.

What cl-waffe cannot currently do on its own:

1. FP16 Operation (It's important for LLMs)
2. Full GPU acceleration

mgl-mat on which cl-waffe mainly depends provides [Facet API](https://github.com/melisgl/mgl-mat#x-28MGL-MAT-3A-40MAT-FACET-API-20MGL-PAX-3ASECTION-29), and The `Facet API` enables the mgl-mat array to be accessed from CL array. That is, (according to my benchmark, there is some overhead but), existing other matrix operation libraries could be utilised. This is why cl-waffe depends on mgl-mat.

True, cl-waffe works like in the relationship shown in this flow.

```
[cl-waffe]->[mgl-mat]->[Any library in Common Lisp]
                     ->[My implementation]
		     ->[OpenBLAS]
		     ->[CUDA]
		     ....
```

## cl-waffe as a deep learning framework

We wrap such existing libraries and define forward and backward propagation via the macro `defnode`, thus enabling automatic differentiation.

I think this is not an ideal situation because `array-to-mat` and `numcl:transpose` is both creating copies, but this is simultaneously good example to show what I want to do:

```lisp
(defnode TransposeOriginalTensor (shape)
  :parameters ((prev-shape nil)
               (shape shape :type cons))
  :forward ((x)
	    (setf (self prev-shape) (!shape x))
	    (with-facet (array ((value x) 'array :direction :input))
	      ; In defnode, it is not always necessary to use the cl-waffe API.
	      ; With regard to this example, it defines a transpose with numcl's API.
	      (sysconst (array-to-mat (numcl:transpose array)))))
  :backward ((dy)
	     (list (!transpose1 dy (self prev-shape)))))

(defun !transpose1 (tensor &rest dims)
  ; defined nodes are called with call
  (call (TransposeOriginalTensor dims) tensor))

(!transpose (!randn `(10 3)))

;#Const(((-0.21... -1.92... ~ 0.560... -0.90...)        
;                 ...
;        (0.580... 0.197... ~ -0.86... 0.765...)) :dtype :float :shape (3 10) :backward <Node: TRANSPOSEORIGINALTENSOR{W2995}>)
```

This is the basic idea behind cl-waffe's automatic differentiation.

These nodes are combined to define a `model` (via defmodel macro). Also, the model has `trainable parameters` and they're optimized by optimizers, defined by `defoptimizer`.

(P.S.: I know not everyone likes this Chainer-like system (define-by-run), so I'm thinking of providing Keras-like APIs.)

Anyway, these features have been developed as **extensible APIs** and do not need to be known by everyone.

There are still **very few standard implementations of NNs(As of this writing, only supports RNN/Linear and so on...)/**, as I think it is important to get the computational fundamentals in place before implementing various deep learning methods. (but contributions are welcome!)

# News

- (2023/4/4) The [documentations](https://hikettei.github.io/cl-waffe-docs/docs/overview.html) are being rewritten. (but only half finished. ><)
- (2023/4/2) The entire cl-waffe code, (especially, `call` and `call-backward`), has been optimised. https://github.com/hikettei/cl-waffe/pull/120 . Accordingly, the API has changed significantly. (e.g.: the function `call` is now a macro.) However, **The Document isn't up-to-date**.

- (2023/03/26) I published the benchmark compared to Numpy/PyTorch. Available at [Here](https://github.com/hikettei/cl-waffe/blob/main/benchmark/Result.md). (Not quite up to my goal.) cl-waffe should peform better... however I guess there's a room to optimize in the cl-waffe's codes...

# Documents

[Documentation](https://hikettei.github.io/cl-waffe-docs) is available.

Also, I started writing [Tutorials(Written in Japanese)](https://github.com/hikettei/cl-waffe/tree/main/tutorials/jp).

As of this writing, available tutorials are written in Japanese and their writing continues, but eventually, I'm willing to complete and translate them into English. So don't worry if you don't speak Japanese.

# MNIST Example

See also: [Document](https://hikettei.github.io/cl-waffe-docs/docs/mnist-tutorial.html)

cl-waffe aimed to reduce the amount of total code written.

```lisp
; Full Code is in ./examples/mnist.lisp

(defmodel MLP (activation)
  :parameters ((layer1   (denselayer (* 28 28) 512 t activation))
	       (layer2   (denselayer 512 256 t activation))
	       (layer3   (linearlayer 256 10 t)))
  :forward ((x)
	    (with-calling-layers x
	      (layer1 x)
 	      (layer2 x)
	      (layer3 x))))

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
 :predict ((x) (call (model) x)))
 
(let ((model (MLPTrainer :relu 1e-3)))
  (step-model model (!randn `(10 784))))
```

# Features

I've only just started developing it, so I'm trying out a lot of features by hand. (That is some features below may well work, some may not.)

As of this writing:
- Broadcasting
- Destructive APIs with a Simple Rule.
- Useful APIs like Numpy/PyTorch
- Automatic Differentiation
- Useful Lazy-Evaluation System
- Eazy to optimize.
- Extensible APIs
- Switchable Backends

## Broadcasting

See also: [Document](https://hikettei.github.io/cl-waffe-docs/docs/using-tensor.html#broadcasting)

cl-waffe has a broadcasting operations like other frameworks.

```lisp
(setq a (!randn `(1 100 200)))
;#Const((((1.900... -0.70... ~ 0.609... 1.397...)         
;                   ...
;         (0.781... 1.735... ~ -1.01... 0.152...))) :mgl t :shape (1 100 200))
(setq b (!randn `(100 100 200)))
;#Const((((-1.21... 0.823... ~ 2.001... -0.21...)         
;                   ...
;         (-0.34... 0.441... ~ -0.07... -0.38...))        
;                 ...
;        ((1.627... 1.127... ~ 0.705... 0.798...)         
;                   ...
;         (0.070... 1.883... ~ 1.850... -0.47...))) :mgl t :shape (100 100 200))

(time (!add a b))
;Evaluation took:
;  0.003 seconds of real time
;  0.002999 seconds of total run time (0.002799 user, 0.000200 system)
;  100.00% CPU
;  6,903,748 processor cycles
;  8,163,776 bytes consed
  
;#Const((((0.689... 0.115... ~ 2.611... 1.183...)         
;                   ...
;         (0.435... 2.177... ~ -1.08... -0.23...))        
;                 ...
;        ((3.528... 0.419... ~ 1.315... 2.195...)         
;                   ...
;         (0.851... 3.619... ~ 0.839... -0.32...))) :mgl t :shape (100 100 200))
```

## Destructive APIs with a Simple Rule.

See also: [Document](https://hikettei.github.io/cl-waffe-docs/docs/using-tensor.html#compute-tensors-in-a-destructive-way)

Internally, Just add to your code `(!allow-destruct tensor)`, cl-waffe regards the tensor as unnecessary and destruct it. This is how implemented destructive operations are.

```lisp
(setq a (!randn `(100 100 100)))
(setq b (!randn `(100 100 100)))

(time (!!add a b))
;Evaluation took:
;  0.000 seconds of real time
;  0.000662 seconds of total run time (0.000605 user, 0.000057 system)
;  100.00% CPU
;  1,422,578 processor cycles
;  0 bytes consed
  
;#Const((((-1.47... 1.016... ~ -1.29... -1.71...)         
;                   ...
;         (2.276... 0.878... ~ -1.35... 0.466...))        
;                 ...
;        ((1.712... 1.318... ~ 0.213... 1.262...)         
;                   ...
;         (1.084... -0.18... ~ -1.42... 0.552...))) :mgl t :shape (100 100 100))
```

## Useful APIs like Numpy/PyTorch.

See also: [Document](https://hikettei.github.io/cl-waffe-docs/docs/cl-waffe.html), Here's the list of all APIs in cl-waffe.

Here's API like `SliceTensor` in Numpy/PyTorch. Of course, they're differentiable.

However, in practical, using offsets (in lisp, we call it displacement) will perform better. (e.g.: setting batch, applying word-by-word processing in RNN). so it is just extra.

```lisp
(setq a (!randn `(100 100 100)))
;#Const((((-1.45... -0.70... ~ -0.87... -0.52...)         
;                   ...
;         (0.655... -1.47... ~ -2.10... -1.79...))        
;                 ...
;        ((-0.28... -1.75... ~ -1.28... 0.381...)         
;                   ...
;         (-0.55... -0.53... ~ 0.421... -0.13...))) :mgl t :shape (100 100 100))

(time (!aref a 0 0 0))
;Evaluation took:
;  0.000 seconds of real time
;  0.000163 seconds of total run time (0.000135 user, 0.000028 system)
;  100.00% CPU
;  235,060 processor cycles
;  0 bytes consed
  
;#Const((((-1.45...))) :mgl t :shape (1 1 1))

(time (!aref a t 0 0))
;Evaluation took:
;  0.000 seconds of real time
;  0.000477 seconds of total run time (0.000455 user, 0.000022 system)
;  100.00% CPU
;  963,246 processor cycles
;  98,256 bytes consed
  
;#Const((((-1.45...))        
;                 ...
;        ((-0.28...))) :mgl t :shape (100 1 1))

(time (!aref a '(0 3) '(10 -1) t))

;Evaluation took:
;  0.001 seconds of real time
;  0.001489 seconds of total run time (0.001445 user, 0.000044 system)
;  100.00% CPU
;  3,518,516 processor cycles
;  322,144 bytes consed
  
;#Const((((-0.10... 0.226... ~ -1.68... 0.662...)         
;                   ...
;         (-0.14... 1.239... ~ -0.90... -0.60...))        
;                 ...
;        ((-0.97... 1.588... ~ 0.558... -1.79...)         
;                   ...
;         (-0.80... -1.50... ~ -1.11... -0.21...))) :mgl t :shape (3 89 100))

(time (!aref a t t t))
;Evaluation took:
;  0.024 seconds of real time
;  0.024426 seconds of total run time (0.024367 user, 0.000059 system)
;  100.00% CPU
;  56,193,050 processor cycles
;  12,675,952 bytes consed
  
;#Const((((-1.45... -0.70... ~ -0.87... -0.52...)         
;                   ...
;         (0.655... -1.47... ~ -2.10... -1.79...))        
;                 ...
;        ((-0.28... -1.75... ~ -1.28... 0.381...)         
;                   ...
;         (-0.55... -0.53... ~ 0.421... -0.13...))) :mgl t :shape (100 100 100))

(setq b (!ones `(100 3)))
(time (setf (!aref a '(0 3)) b))
;Evaluation took:
;  0.001 seconds of real time
;  0.001312 seconds of total run time (0.001274 user, 0.000038 system)
;  100.00% CPU
;  2,898,956 processor cycles
;  262,048 bytes consed
;#Const((((1.0 1.0 ~ -0.87... -0.52...)         
;                   ...
;         (1.0 1.0 ~ -2.10... -1.79...))        
;                 ...
;        ((-0.28... -1.75... ~ -1.28... 0.381...)         
;                   ...
;         (-0.55... -0.53... ~ 0.421... -0.13...))) :mgl t :shape (100 100 100))
```


## Automatic Differentiation

See also: [Document](https://hikettei.github.io/cl-waffe-docs/docs/using-tensor.html#basic-tensor-operations)

Once forward is defined, backward is also automatically defined. This feature is indispensable for Deep Learning Framework. Of course it is available.

```lisp
(setq a (parameter (!randn `(10 10))))
(setq b (parameter (!randn `(10 10))))
(setq c (parameter (!randn `(10))))


(setq z (!sum (!add (!mul a b) c)))

(time (backward z))
;Evaluation took:
;  0.001 seconds of real time
;  0.001469 seconds of total run time (0.001344 user, 0.000125 system)
;  100.00% CPU
;  3,239,008 processor cycles
;  130,048 bytes consed
  
;NIL

(print (const (grad a)))
;#Const(((0.004... -0.00... ~ 0.004... 5.721...)
;                 ...
;        (0.001... 5.919... ~ 7.748... -0.00...)) :mgl t :shape (10 10))
(print (const (grad b)))
;#Const(((0.004... -0.00... ~ 0.004... 5.721...)
;                 ...
;        (0.001... 5.919... ~ 7.748... -0.00...)) :mgl t :shape (10 10))
(print (const (grad c)))
;#Const((0.01 0.01 ~ 0.01 0.01) :mgl t :shape (10))
```

## Useful Lazy-Evaluation System

See also: [Document](https://hikettei.github.io/cl-waffe-docs/docs/using-tensor.html#lazy-evaluation)

cl-waffe provides zero-cost transpose by using lazy-evaluation.

Just use `!transpose` before `!matmul`, `!matmul` automatically recognises it and the retuend tensor is applied `transpose`.

Here's `!transpose1` for the case when you just want a transposed tensor.

The lazy-evaluated tensors are evaluated via function `(value tensor)`. Once this function called. the content of tensor is fulfilled with a new evaluated matrix. Don't worry, `(value tensor)` are scattered all over the place in cl-waffe's code, so no additional codes are required.

```lisp
(setq a (!randn `(10 3)))
;#Const(((0.411... 0.244... 2.828...)        
;                 ...
;        (-1.26... -1.41... 0.821...)) :mgl t :shape (10 3))

(!transpose a)
;#Const(#<FUNCTION (LABELS CL-WAFFE.BACKENDS.MGL::LAZYTRANSPOSE :IN CL-WAFFE.BACKENDS.MGL::LAZY-EVAL-TRANSPOSE) {100CA0F5EB}>)

; Generally, using delayed evaluation does not require additional new code.
(!add * 1)
;#Const(((1.411... 0.862... ~ 1.590... -0.26...)        
;                 ...
;        (3.828... 0.582... ~ -0.57... 1.821...)) :mgl t :shape (3 10))

; Using !transpose is much faster for !matmul (even when the tensors are 3d/4d...).
(time (!matmul a (!transpose a)))
;Evaluation took:
;  0.000 seconds of real time
;  0.000107 seconds of total run time (0.000101 user, 0.000006 system)
;  100.00% CPU
;  147,434 processor cycles
;  0 bytes consed
  
;#Const(((8.227... -1.29... ~ -4.10... 1.458...)        
;                 ...
;        (1.458... 0.180... ~ -2.59... 4.273...)) :mgl t :shape (10 10))

; Compared to !transpose1...
(time (!matmul a (!transpose1 a)))
;Evaluation took:
;  0.178 seconds of real time
;  0.176052 seconds of total run time (0.120406 user, 0.055646 system)
;  98.88% CPU
;  4 forms interpreted
;  773 lambdas converted
;  410,887,630 processor cycles
;  25,051,200 bytes consed
  
;#Const(((8.227... -1.29... ~ -4.10... 1.458...)        
;                 ...
;        (1.458... 0.180... ~ -2.59... 4.273...)) :mgl t :shape (10 10))

; PS (2023/05/26). The lazy-evaluated tensors have been modified to display more elegant.
(print a) ; #Const(<Transposed Tensor> :shape (10 10) :backward <Node: TRANSPOSETENSOR{W2126}>)
```

```lisp
(setq a (!randn `(10 10)))
;#Const(((0.570... -0.13... ~ 0.217... 0.862...)        
;                 ...
;        (-0.12... 0.384... ~ -0.25... -0.91...)) :mgl t :shape (10 10))

(setq lazy-evaluated-a (!transpose a))
;#Const(#<FUNCTION (LABELS CL-WAFFE.BACKENDS.MGL::LAZYTRANSPOSE :IN CL-WAFFE.BACKENDS.MGL::LAZY-EVAL-TRANSPOSE) {100E48135B}>)

(print lazy-evaluated-a)
;#Const(#<FUNCTION (LABELS CL-WAFFE.BACKENDS.MGL::LAZYTRANSPOSE :IN CL-WAFFE.BACKENDS.MGL::LAZY-EVAL-TRANSPOSE) {100E48135B}>)

; value will accept and evaluated lazy-evaluated tensor.
(value lazy-evaluated-a)

(print lazy-evaluated-a)
;#Const(((0.570... 1.228... ~ 0.050... -0.12...)        
;                 ...
;        (0.862... -0.82... ~ 1.360... -0.91...)) :mgl t :shape (10 10))
```

## Eazy to optimize

For more detail: [defnode and call](https://hikettei.github.io/cl-waffe-docs/docs/tutorials.html#defnode-and-call)

Nodes are described in a clear and highly functional notation:

```lisp
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
  :backward ((dy) (list dy dy)))
```

It can be easily inlined via the macro `call`.

```lisp
(macroexpand `(call (ScalarAdd) (const 1.0) (const 1.0)))
```

```lisp
(LOCALLY
 (DECLARE (OPTIMIZE (SPEED 3) (SAFETY 1))
          (INLINE call-scalaradd-forward-mgl))
 (call-scalaradd-forward-mgl (SCALARADD) (CONST 1.0) (CONST 1.0)))
```

## Extensible APIs

See also: [Document](https://hikettei.github.io/cl-waffe-docs/docs/extend-library.html)

cl-waffe's features are based on these macro:

- `defmodel` (describes forward process and parameters that to be optimized.)
- `defnode`  (describes forward process and backward process, that not necessary to use cl-waffe/mgl-mat, you can use libraries you like.)
- `deftrainer` (describes the predict/training process. (e.g.: criterion, optimizer's setting), which contributes to reduce the amount of total codes.)
- `defoptimizer` (describes the optimizing function)
- `defdataset` (describes how the dataset is itearted.)

True, almost implementations are using it (See also: `./source/optimizers/optimizers.lisp`, or `./source/operators.lisp`). In the same way, All macros are exported, and users can make extensions of the framework as required. 

(The codes below is using mgl-mat and numcl. For details about with-facet, numcl: [with-facet(mgl-mat)](https://github.com/melisgl/mgl-mat#x-28MGL-MAT-3A-40MAT-FACET-API-20MGL-PAX-3ASECTION-29), [numcl](https://github.com/numcl/numcl))

```lisp
; in ./source/operators.lisp at 202th line

(defnode TransposeOriginalTensor (shape)
  :optimize t
  :parameters ((prev-shape nil) (shape shape))
  :forward ((x)
	    (setf (self prev-shape) (!shape x))
	    (with-facet (array ((value x) 'array :direction :input))
	      ; In defnode, it is not always necessary to use the cl-waffe API.
	      ; With regard to this example, it defines a transpose with numcl's API.
	      (sysconst (array-to-mat (numcl:transpose array)))))
  :backward ((dy)
	     (list (!transpose1 dy (self prev-shape)))))

(defun !transpose1 (tensor &rest dims)
  ; defined nodes are called with call
  (call (TransposeOriginalTensor dims) tensor))
  
; in ./source/optimizers/optimizers.lisp at 4th line

(defoptimizer SGD (params &key (lr 1e-3))
  :parameters ((params params) (lr lr))
  :update (()
	   (dotimes (i (hash-table-count (self params)))
	     (copy! (data (!sub (gethash i (self params))
					   (!mul (self lr) (grad (gethash i (self params))))))
			  (data (gethash i (self params)))))))
```

## Switchable Backends

See also: [Documentation](https://hikettei.github.io/cl-waffe-docs/docs/using-tensor.html#backends)

It is allowed to redefine the original node in cl-waffe. Such nodes are managed by using `backend`.

`define-node-extension` is available to extend the existing nodes.

```lisp
; in ./t/node-extension.lisp
(define-node-extension cl-waffe::AddTensor
  :backend :my-extension
  :forward ((x y)
            (const (+ 100 100)))
  :backward ((dy)
             (list dy dy)))

(defun operate-in-mgl ()
  (with-backend :mgl
    (= (data (!add 1 1)) 2)))

(defun operate-in-extension ()
  (with-backend :my-extension
    (= (data (!add 1 1)) 200)))

(defun operate-restart-test () ; if the operation doesn't exists...
  (with-backend :does-not-exists
    (= (data (!add 1 1)) 2)))
```

# Install

It is recommended to install following in advance:

1. [SBCL](https://www.sbcl.org/)
2. [Roswell](https://github.com/roswell/roswell) (If you're new to Common Lisp, I recommend you to install it first.)
3. [Lake](https://github.com/takagi/lake)

cl-waffe is available in one of the following ways:

### Install via Github
For Example:
```shell
$ git clone git@github.com:hikettei/cl-waffe.git
$ cd cl-waffe
$ sbcl
* (load "cl-waffe.asd")
* (ql:quickload :cl-waffe) ; all is done!
```

### Install via Roswell

```shell
$ ros install hikettei/cl-waffe
$ ros run
* (ql:quickload :cl-waffe)
```

### Install via Ultralisp

[ultralisp](https://ultralisp.org/) dist is available.

# Contributing

Please read [CONTRIBUTING.md](./CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us. Don't hesitate to send me issues if you have any trouble!

### Prerequisites

1. SBCL
2. Roswell
3. Lakefile
4. Quicklisp

### Running the tests

```shell
$ lake test
```

should work.

## Lakefile

[Lakefile](https://github.com/leanprover/lake) is available at github repository. (Also it requires [Roswell](https://github.com/roswell/roswell))

```shell
$ lake
Usage: lake [command]

Tasks:
  test                     Operate tests
  benchmark                Start Benchmarking
  benchmark-python         Start Benchmarking with cl-waffe and numpy/pytorch.
  gendoc                   Generating Documentations
  example:install          Install training data for examples
  example:mnist            Run example model with MNIST
  example:rnn              Run example model with Seq2Seq
```

# Run MNIST With Roswell


```shell
$ cd examples
$ sh install.sh
$ cd ..
$ ./run-test-model.ros mnist
```

or

```shell
$ lake example:install
$ lake example:fnn
```

should work. `lake example:mnist` is also OK.

# Currently Problems/Todo
As of writing, I'm working on:

- ~~破壊的代入のサポート(Support more destructive operations)~~(Done)
- Neural Networkの追加 (Add cl-waffe.nn models)
- RNNs are too much slower than PyTorch...
- モデルの保存に対応 (Save and restore trained models.)
- グラフの表示に対応 (Plotting losses and so on)
- 様々なデータ構造を扱えるように (Support more types of data structure)
- 性能向上（メモリ使用量/CPU使用率の観点から
）(In term of cpu-usage rate/memory-usage, cl-waffe has a lot of challenge to performance.)
- CUDAに対応 (Support CUDA)
- 他の処理系で動くか試す (Try on another systems (e.g.: CCL))
- Improving the quality of documentation.

# Goals

- Making cl-waffe a modern and fast framework.
	- Fix: high memory usage
	- Add: More APIs
	- Add: Clear distinction between slow and fast APIs.
	- Add: Simple rules to make it fast and lacklustre and documentations for it
	- Goal: Training Transformer Model
	
- Making cl-waffe practical
	- Support: cl-jupyter, any plotting library, matplotlib, etc...
	- Support: CUDA with Full Performance!
	- Add: Mathematical Functions
	- Add: High power APIs (such features are rooted in !aref, broadcasting. they need to be optimized more)
	- Add: DataLoader like PyTorch
	- Add: Save and Restore Models, (Compatible with PyTorch if possible...)
	
- Go faster cl-waffe
	- Support: more parallelized operators
	- Keep whole codes abstracted and extensible
	- Apply full optimisation when some functionality is reached enough.
	- More benchmarks are needed and put it all in a table somewhere.

I love Common Lisp very much and there are many excellent libraries for numerical operations with great ideas.

However, I know I'm really reckless, but the one I want to make is:
- Making full use of Common Lisp's nice features.
- I want to have a range of functions comparable to Python's frameworks.
- Simple/Compact notations and APIs

Having started on 2022/12/26, this project will take a long time before these features are realised.


Does anyone have any ideas? Please share with me on issues!

Also, bug reports and more are welcome!

# Acknowledgements

- The author of [mgl-mat](https://github.com/melisgl/mgl-mat), and anyone who has contributed to it. 
- To all those who gave me ideas, helps and knowledgement.

# Author

hikettei
- [Twitter](https://twitter.com/ichndm) 
- [Github](https://github.com/hikettei)
- Discord: rulia🌙#5298

# Environment


- SBCL
	- it is recommended to use SBCL, I've not tested on others
