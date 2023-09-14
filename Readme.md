[![CI](https://github.com/hikettei/cl-waffe/actions/workflows/ci.yml/badge.svg)](https://github.com/hikettei/cl-waffe/actions/workflows/ci.yml)

> ⚠️ This repository is abandoned by me since the development has complemently moved to [cl-waffe2](https://github.com/hikettei/cl-waffe2).
> 
> cl-waffe2 provides: by far the fastest, systematic, easy to optimize, customizable, and environment- and device- independent abstract matrix operations while cl-waffe do not.
>
> So it is recommended to check out cl-waffe2 instead of this project.

# cl-waffe

cl-waffe is a deep learning framework with modern APIs for Common Lisp, based on [mgl-mat](https://github.com/melisgl/mgl-mat). 

# TOC

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [What cl-waffe does?](#what-cl-waffe-does)
  - [cl-waffe as a matrix operation library](#cl-waffe-as-a-matrix-operation-library)
  - [cl-waffe as a deep learning framework](#cl-waffe-as-a-deep-learning-framework)
- [News](#news)
- [MNIST Example](#mnist-example)
- [Features](#features)
  - [Useful and high-level API](#useful-and-high-level-api)
    - [Broadcasting](#broadcasting)
    - [Highly functional Aref](#highly-functional-aref)
    - [Rich APIs](#rich-apis)
  - [Automatic Differentiation](#automatic-differentiation)
  - [Eazy to optimize.](#eazy-to-optimize)
    - [Fully Inlined Nodes](#fully-inlined-nodes)
    - [Lazy-Evaluation](#lazy-evaluation)
  - [Extensible APIs](#extensible-apis)
- [Install](#install)
    - [Requirements](#requirements)
    - [Install via Github](#install-via-github)
    - [Install via Roswell](#install-via-roswell)
    - [Install via Ultralisp](#install-via-ultralisp)
- [Contributing](#contributing)
    - [Running the tests](#running-the-tests)
    - [Lakefile](#lakefile)
- [Trying cl-waffe with Example Models](#trying-cl-waffe-with-example-models)
- [Acknowledgements](#acknowledgements)
- [Author](#author)
- [Environment](#environment)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


# What cl-waffe does?

cl-waffe is the two sides of the coin:
1. As a **matrix operation library**.
2. As a **deep learning framework**.

## cl-waffe as a matrix operation library

For the first purpose, cl-waffe aims to wrap existing Common Lisp matrix operation libraries with Numpy/PyTorch like APIs, reducing overheads between cl-waffe and these libraries.

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

For more details: [defnode and call](https://hikettei.github.io/cl-waffe-docs/docs/tutorials.html#defnode-and-call)

The macros **defnode** and **call** serves as a key component of cl-waffe. In designing deep learning models, incorporating object-oriented programming can lead to more consice descriptions. Although Common Lisp has a powerful framework: CLOS and Closer-MOP, but I think its computational speed strongly depends on what common lisp implementation to use. (e.g.: SBCL/Clozure CL...) Thus, by using only defstruct and defun for defining the computation nodes and wrapping them with macros, (defnode) and (call), I have reduced the overhead associated with the process. This example shows how to define ScalarAdd Node.

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

```lisp
(time (call (ScalarAdd) (const 1.0) (const 1.0))) ; via cl-waffe
;#Const(2.0 :dtype SINGLE-FLOAT :backward <Node: SCALARADD{W924}>)
;Evaluation took:
;  0.000 seconds of real time
;  0.000005 seconds of total run time (0.000005 user, 0.000000 system)
;  100.00% CPU
;  11,084 processor cycles
;  0 bytes consed
  
(time (+ 1.0 1.0)) ; CL Native
;2.0
;Evaluation took:
;  0.000 seconds of real time
;  0.000001 seconds of total run time (0.000000 user, 0.000001 system)
;  100.00% CPU
;  422 processor cycles
;  0 bytes consed
```

Nodes called by the macro `(call) `are fully inlined, (like CL's `inline-generic-function`, `static-dispatch`). Considering ScalarAdd builds computation node in addition to summing up the arguments, these overheads are enough small.

# News

- (2023/4/4) The [documentations](https://hikettei.github.io/cl-waffe-docs/docs/overview.html) are being rewritten. (but only half finished. ><)
- (2023/4/2) The entire cl-waffe code, (especially, `call` and `call-backward`), has been optimised. https://github.com/hikettei/cl-waffe/pull/120 . Accordingly, the API has changed significantly. (e.g.: the function `call` is now a macro.) However, **The Document isn't up-to-date**.

- (2023/03/26) I published the benchmark compared to Numpy/PyTorch. Available at [Here](https://github.com/hikettei/cl-waffe/blob/main/benchmark/Result.md). (Not quite up to my goal.) cl-waffe should peform better... however I guess there's a room to optimize in the cl-waffe's codes...

# MNIST Example

See also: [Document](https://hikettei.github.io/cl-waffe-docs/docs/mnist-tutorial.html)

This example demonstrates a three-layer MLP implemented using cl-waffe.

With the help of cl-waffe, you can define models consisely like this:

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
  (step-model model (!randn `(10 784)) (!ones `(10 10))))
```

# Features

As of this writing:

- Useful and high-level API
- Automatic Differentation
- Eazy to optimize
- Extensible APIs

## Useful and high-level API

The standard cl-waffe API includes features like these, which are also supported by other Python libraries.

- Broadcasting
- Highly functional Aref
- Rich API

### Broadcasting

```lisp
(setq a (!randn `(100 100 100)))
(setq b (!randn `(100 1)))

(time (!add a b))
;Evaluation took:
;  0.004 seconds of real time
;  0.004748 seconds of total run time (0.003399 user, 0.001349 system)
;  125.00% CPU
;  11,061,940 processor cycles
;  4,190,448 bytes consed
  
;#Const((((-1.25... -0.46... ~ 0.265... -0.37...)         
;                   ...
;         (0.675... -0.77... ~ -1.50... -1.22...))        
;                 ...
;        ((-0.72... -0.25... ~ 1.381... 0.727...)         
;                   ...
;         (0.198... 0.178... ~ -2.18... -1.40...))) :dtype :float :shape (100 100 100) :backward <Node: BROADCASTINGADDTENSOR{W90085}>)
```

### Highly functional Aref

```lisp
(setq a (!init-with `(1000 1000) #'(lambda (x) x)))
;#Const(((0.0 1.0 ~ 998.0... 999.0...)        
;                 ...
;        (99900... 99900... ~ 99999... 99999...)) :dtype :float :shape (1000 1000) :backward NIL)

(time (!aref a 0 t))
;Evaluation took:
;  0.000 seconds of real time
;  0.000078 seconds of total run time (0.000078 user, 0.000000 system)
;  100.00% CPU
;  177,232 processor cycles
;  0 bytes consed
  
;#Const(((0.0 1.0 ~ 998.0... 999.0...)) :dtype :float :shape (1 1000) :backward <Node: AREFTENSOR{W90108}>)

(time (!aref a `(2 -1) t))
;Evaluation took:
;  0.007 seconds of real time
;  0.007651 seconds of total run time (0.006475 user, 0.001176 system)
;  114.29% CPU
;  17,779,036 processor cycles
;  4,909,136 bytes consed
  
;#Const(((2000.... 2001.... ~ 2998.... 2999....)        
;                 ...
;        (99800... 99800... ~ 99899... 99899...)) :dtype :float :shape (997 1000) :backward <Node: AREFTENSOR{W90109}>)

(setf (!aref a '(0 10) t) (!ones `(10)))
;#Const(((1.0 1.0 ~ 1.0 1.0)        
;                 ...
;        (99900... 99900... ~ 99999... 99999...)) :dtype :float :shape (1000 1000) :backward <Node: SETFAREFTENSOR{W90130}>)
```

### Rich APIs

There are many operations available in cl-waffe, and I am going to continue expanding them in the future.

See also: [Document](https://hikettei.github.io/cl-waffe-docs/docs/cl-waffe.html)

## Automatic Differentiation

define-by-run style:

```lisp
(setq a (parameter (!randn `(3 3))))
(setq b (parameter (!randn `(3 3))))
(setq c (parameter (!randn `(3 3))))

(setq z (!sum (!add (!mul a b) c)))

(backward z)

(grad a)
(grad b)
(grad c)
```

## Eazy to optimize.

### Fully Inlined Nodes

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

### Lazy-Evaluation

Zero-cost transpose is achieved through the use of lazy evaluation.

```lisp
(setq a (!randn `(100 20)))
(setq b (!randn `(100 20)))

(!transpose a)
;#Const(<Transposed Tensor> :shape (20 100) :backward <Node: TRANSPOSETENSOR{W90135}>)

(time (!matmul (!transpose a) b))
;Evaluation took:
;  0.001 seconds of real time
;  0.000312 seconds of total run time (0.000085 user, 0.000227 system)
;  0.00% CPU
;  4,329,458 processor cycles
;  0 bytes consed
  
;#Const(((5.946... -6.45... ~ -14.0... 10.40...)        
;                 ...
;        (8.740... 6.130... ~ -6.76... -3.12...)) :dtype :float :shape (20 20) :backward <Node: MATMULTENSOR{W90165}>)
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

Also, it is allowed to redefine the original node in cl-waffe. Such nodes are managed by using `backend`.

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
```

# Install

### Requirements

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

### Running the tests

```shell
$ lake test
```

should work.

### Lakefile

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

# Trying cl-waffe with Example Models

```shell
$ lake example:install
$ lake example:fnn
```

If Lake isn't available in your environment, try this:

```shell
$ cd examples
$ sh install.sh
$ cd ..
$ ./run-test-model.ros fnn
```

either of them should work. `lake example:mnist` is also OK.

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
