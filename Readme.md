
# cl-waffe, a deep-learning library, that is mgl-based and extensible and easy to use.

This package is still under development.

I'm building this package for my studying data science. So the features could be insufficient to practice.

The future goal is that easy to use, not speed, since building neural networks in common lisp is tend to be complicated.

However, cl-waffe is at least faster than PyTorch in the benchmark following, using excellent libraries such as mgl-mat, and numcl.

# Benchmark

Coming soon...

# MNIST Example

```lisp

; full code is in examples/mnist.lisp


; define cl-waffe model.  it can be accessed from a trainer-object defined by deftrainer
(defmodel MLP (activation)
  :parameters ((layer1 (cl-waffe.nn:denselayer (* 28 28) 512 T activation))
	       (layer2 (cl-waffe.nn:denselayer 512 256 T activation))
	       (layer3 (cl-waffe.nn:linearlayer 256 10 T)))
  :forward ((x)
	    (call (self layer3)
		  (call (self layer2)
			(call (self layer1) x)))))


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


(defun demo ()
  (multiple-value-bind (datamat target)
      (read-data "examples/tmp/mnist.scale" 784 10 :most-min-class 0)
    (defparameter mnist-dataset datamat)
    (defparameter mnist-target target))

  (multiple-value-bind (datamat target)
      (read-data "examples/tmp/mnist.scale.t" 784 10 :most-min-class 0)
    (defparameter mnist-dataset-test datamat)
    (defparameter mnist-target-test target))

  (format t "Training: ~a" (!shape mnist-dataset))
  (format t "Valid   : ~a" (!shape mnist-target))
  (print "")


  (setq trainer (MLPTrainer :relu 1e-4))

  (setq train (MnistData mnist-dataset mnist-target 100))
  (setq test (MnistData mnist-dataset-test mnist-target-test 100))

  (sb-sprof:start-profiling)
  (time (train trainer train :max-iterate 600 :epoch 10 :batch-size 100 :valid-dataset test :verbose t :random t))
  (sb-profile:report))

```

# Features

Full Version is Coming soon...

As you can see from `./source/optimizers/optimizers.lisp`, or `./source/operators.lisp`,  the features like `defnode`, `defoptimizer` is exported for users.
Here's examples.

```lisp
; in ./source/operators.lisp at 17th line
(defnode AddTensor nil
  :parameters nil
  :forward  ((x y)
	     (with-searching-calc-node :add x y))
  :backward ((dy) (list dy dy)))

; in ./source/optimizers/optimizers.lisp at 4th line

(defoptimizer SGD (params &key (lr 1e-3))
  :parameters ((params params) (lr lr))
  :update (()
	   (dotimes (i (hash-table-count (self params)))
	     (mgl-mat:copy! (data (!sub (gethash i (self params))
					(!mul (self lr) (grad (gethash i (self params))))))
			    (data (gethash i (self params)))))))
```

# Run MNIST With Roswell

```
$ cd examples
$ sh install.sh
$ cd ..
$ ./run-test-model.ros mnist
```

# Documents

Coming soon...

# Workload/Todo

・Speed up whole code

・Implement the features for NLP, (e.g. Embedding, LSTM...)

・Current definition of adam is too slow to use, so i need to optimize

・More functions for mathmatics.

・3d mat operations

・save and restore models

・Automatic convert from non-destructive operation to !modify call in forward process.

# Tutorials

It is better to run this command in advance, since training a model requires a lot of memory.

```
$ ros config set dynamic-space-size 4gb
```

# Author

hikettei (Twitter: @ichndm, github:@hikettei)

### memos

; nodeのパラメーターの初期値にnil使えないのを覚えておく


## Environment

```
SBCL 2.2.5
macOS Monterey version 12.4
```


# Welcome To cl-waffe!

cl-waffeはCommonLispで実装されたDeeplearningのライブラリです。

cl-waffeでは以下の3つのマクロと2つの拡張用のマクロを用いてネットワークを定義していきます。
