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

- [What cl-waffe does?](#what-cl-waffe-does)
  - [cl-waffe as a matrix operation library](#cl-waffe-as-a-matrix-operation-library)
  - [cl-waffe as a deep learning framework](#cl-waffe-as-a-deep-learning-framework)
- [News](#news)
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
  (step-model model (!randn `(10 784))))
```

# Features

As of this writing:

- Useful and high-level APIs
- Automatic Differentation
- Eazy to optimize
- Extensible APIs

## Useful and high-level APIs


See also: [Document](https://hikettei.github.io/cl-waffe-docs/docs/using-tensor.html#broadcasting)
See also: [Document](https://hikettei.github.io/cl-waffe-docs/docs/using-tensor.html#compute-tensors-in-a-destructive-way)
See also: [Document](https://hikettei.github.io/cl-waffe-docs/docs/cl-waffe.html)

broadcasting/aref/destrucitve

...

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


## Eazy to optimize.

- lazy-eval
- inline

See also: [Document](https://hikettei.github.io/cl-waffe-docs/docs/using-tensor.html#lazy-evaluation)
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

(No Lake but Roswell Ver)
```shell
$ cd examples
$ sh install.sh
$ cd ..
$ ./run-test-model.ros mnist
```

(Roswell and Lake ver)
```shell
$ lake example:install
$ lake example:fnn
```

should work. `lake example:mnist` is also OK.

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
