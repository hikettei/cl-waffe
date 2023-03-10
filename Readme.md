
# cl-waffe
[![CI](https://github.com/hikettei/cl-waffe/actions/workflows/ci.yml/badge.svg)](https://github.com/hikettei/cl-waffe/actions/workflows/ci.yml)
![cl-waffe](https://hikettei.github.io/cl-waffe-docs/docs/cl-waffe-logo.png)

**This package is still under development, and its features are far from practical.**

cl-waffe is a deep learning framework with modern APIs for Common Lisp.

Having not GPUs, I can't test my framework on cuda ><. CUDA support is a little further along.

# Documents

[Documentation](https://hikettei.github.io/cl-waffe-docs) is available.

# TOC

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [MNIST Example](#mnist-example)
- [Features](#features)
  - [Broadcasting.](#broadcasting)
  - [Destructive APIs with a Simple Rule.](#destructive-apis-with-a-simple-rule)
  - [Useful APIs like Numpy/PyTorch.](#useful-apis-like-numpypytorch)
  - [Automatic Differentiation](#automatic-differentiation)
  - [Useful Lazy-Evaluation System](#useful-lazy-evaluation-system)
  - [Tracing JIT](#tracing-jit)
  - [Extensible APIs](#extensible-apis)
- [Usage](#usage)
- [Run MNIST With Roswell](#run-mnist-with-roswell)
- [Currently Problems/Todo](#currently-problemstodo)
- [Goals](#goals)
- [Acknowledgements](#acknowledgements)
- [Author](#author)
- [Environment](#environment)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# MNIST Example

See also: [Document](https://hikettei.github.io/cl-waffe-docs/docs/mnist-tutorial.html)

```lisp
; full code is in examples/mnist.lisp


; define cl-waffe model.  it can be accessed from a trainer-object defined by deftrainer
(defmodel MLP (activation)
  :parameters ((layer1 (cl-waffe.nn:denselayer (* 28 28) 512 T activation))
	       (layer2 (cl-waffe.nn:denselayer 512 256 T activation))
	       (layer3 (cl-waffe.nn:linearlayer 256 10 T)))
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


(defdataset Mnistdata (train valid batch-size)
  :parameters ((train train) (valid valid) (batch-size batch-size))
  :forward ((index)
	    (list (!set-batch (self train) index (self batch-size))
		  (!set-batch (self valid) index (self batch-size))))
  :length (() (car (!shape (self train)))))


...

(setq trainer (MLPTrainer :relu 1e-4))

(setq train (MnistData mnist-dataset mnist-target 100))
(setq test (MnistData mnist-dataset-test mnist-target-test 100))

(time (train trainer train :max-iterate 600 :epoch 10 :batch-size 100 :valid-dataset test :verbose t :random t))

```

# Features

As of this writing:
- Broadcasting
- Destructive APIs with a Simple Rule.
- Useful APIs like Numpy/PyTorch
- Automatic Differentiation
- Useful Lazy-Evaluation System
- Tracing JIT
- Extensible APIs

## Broadcasting.

See also: [Document](https://hikettei.github.io/cl-waffe-docs/docs/using-tensor.html#broadcasting)

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

See also: [Document](https://hikettei.github.io/cl-waffe-docs/docs/cl-waffe.html)

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

cl-waffe's lazy-evaluation system doesn't require any additional code.

Just call `(value tensor)` to accept lazy evaluation.

`!transpose` will produce lazy-evaluated tensor, while `!transpose1` will do not.

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

## Tracing JIT

This is still experimental but...

In `(with-jit)` macro, cl-waffe dynamically defines the kernel functions with lazy-evaluation system. (currently only for blas)

```lisp

; In default...

(time (!log (!exp a)))
;Evaluation took:
;  0.000 seconds of real time
;  0.000171 seconds of total run time (0.000130 user, 0.000041 system)
;  100.00% CPU
;  248,100 processor cycles
;  3,232 bytes consed
  
;#Const(((0.570... -0.13... ~ 0.217... 0.862...)        
;                 ...
;        (-0.12... 0.384... ~ -0.25... -0.91...)) :mgl t :shape (10 10))

(defun trace-operate ()
  (with-jit
     (time (const (value (!log (!exp a)))))))

; The first call of trace-operate, it seems slow because cl-waffe traces and compiles code.
(trace-operate)
;Evaluation took:
;  0.000 seconds of real time
;  0.000183 seconds of total run time (0.000122 user, 0.000061 system)
;  100.00% CPU
;  240,442 processor cycles
;  32,512 bytes consed
  
;#Const(((0.570... -0.13... ~ 0.217... 0.862...)        
;                 ...
;        (-0.12... 0.384... ~ -0.25... -0.91...)) :mgl t :shape (10 10))

; However, after the second one, it will be faster.
(trace-operate)
;Evaluation took:
;  0.000 seconds of real time
;  0.000096 seconds of total run time (0.000087 user, 0.000009 system)
;  100.00% CPU
;  187,848 processor cycles
;  0 bytes consed
  
;#Const(((0.570... -0.13... ~ 0.217... 0.862...)        
;                 ...
;        (-0.12... 0.384... ~ -0.25... -0.91...)) :mgl t :shape (10 10))

; P.S.: Back propagation is available even when with-jit is enabled
```


## Extensible APIs

See also: [Document](https://hikettei.github.io/cl-waffe-docs/docs/extend-library.html)

As you can see from `./source/optimizers/optimizers.lisp`, or `./source/operators.lisp`,  the features like `defnode`, `defoptimizer` is exported for users.
Here's examples.

(For details about with-facet, numcl: [with-facet](https://github.com/melisgl/mgl-mat#x-28MGL-MAT-3A-40MAT-FACET-API-20MGL-PAX-3ASECTION-29), [numcl](https://github.com/numcl/numcl))

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

# Usage

Please clone this repository and register it as a local-project or just load `cl-waffe.asd`

This framework is still **incomplete and experimental**, being not yet ready to register with Quicklisp etc..

For Example:
```shell
$ git clone git@github.com:hikettei/cl-waffe.git
$ cd cl-waffe
$ sbcl
* (load "cl-waffe.asd")
* (ql:quickload :cl-waffe) ; all is done!
```

[Lakefile](https://github.com/leanprover/lake) is available. (Also it requires [Roswell](https://github.com/roswell/roswell))

```shell
$ lake
Usage: lake [command]

Tasks:
  test                     Operate tests
  benchmark                Start Benchmarking
  gendoc                   Generating Documentations
  example:install          Install training data for examples
  example:mnist            Run example model with MNIST
  example:rnn              Run example model with Seq2Seq
```

# Run MNIST With Roswell

```
$ cd examples
$ sh install.sh
$ cd ..
$ ./run-test-model.ros mnist
```

# Currently Problems/Todo
As of writing, I'm working on:

- ??????????????????????????????(Support more destructive operations)
- Neural Network????????? (Add cl-waffe.nn models)
- Iteration???Backward???????????? (Improve performance of RNN) (e.g.: the backward of (setf !aref) ...)
- ??????????????????????????? (Save and restore trained models.)
- ??????????????????????????? (Plotting losses and so on)
- ????????????????????????????????????????????? (Support more types of data structure)
- ?????????????????????????????????/CPU????????????????????????
???(In term of cpu-usage rate/memory-usage, cl-waffe has a lot of challenge to performance.)
- CUDA????????? (Support CUDA)
- ????????????????????????????????? (Try on another systems (e.g.: CCL))
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

- The author of [mgl-mat](https://github.com/melisgl/mgl-mat), since the cl-waffe tensor depends on this. (Without it, the cl-waffe's performance was the worst.)
- To all those who gave me ideas, helps and knowledgement.

# Author

hikettei
- [Twitter](https://twitter.com/ichndm) 
- [Github](https://github.com/hikettei)
- Discord: rulia????#5298

# Environment


- SBCL
	- it is recommended to use SBCL, I've not tested on others
