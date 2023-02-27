
# cl-waffe

![cl-waffe](https://hikettei.github.io/cl-waffe-docs/docs/cl-waffe-logo.png)

**This package is still under development, and its features are far from practical.**

cl-waffe is a deep learning framework for Common Lisp.

# Documents

[Documentation](https://hikettei.github.io/cl-waffe-docs) is available.

# MNIST Example

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

Full Version is Coming soon...

As you can see from `./source/optimizers/optimizers.lisp`, or `./source/operators.lisp`,  the features like `defnode`, `defoptimizer` is exported for users.
Here's examples.

```lisp
; in ./source/operators.lisp at 202th line

(defnode TransposeOriginalTensor (shape)
  :optimize t
  :parameters ((prev-shape nil) (shape shape))
  :forward ((x)
	    (setf (self prev-shape) (!shape x))
	    (with-facet (array ((value x) 'array :direction :input))
	      ; In defnode, it is not always necessary to use the cl-waffe API.
	      ; For this example, defining node with numcl's API.
	      (sysconst (array-to-mat (numcl:transpose array)))))
  :backward ((dy)
	     (list (!transpose1 dy (self prev-shape)))))

; in ./source/optimizers/optimizers.lisp at 4th line

(defoptimizer SGD (params &key (lr 1e-3))
  :parameters ((params params) (lr lr))
  :update (()
	   (dotimes (i (hash-table-count (self params)))
	     (copy! (data (!sub (gethash i (self params))
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

# Workload/Todo

・Allowing destructive operations.

・Plotting graphs

・Optimize whole codes, in term of speed and memory space.

・The usage rate of CPU is too low...

・Implement a convenient data loader package and avoid users having to implement it themselves.

・support cuda in some operations

・More functions for mathmatics.

・save and restore models

・Rewrite some nodes with (with-kernel-case)

・CCLで試す

・run-test-model.rosを使いやすく書き直す

・Documentを書く

# Author

hikettei (Twitter: @ichndm, github:@hikettei)


## Environment

```
SBCL 2.2.5
macOS Monterey version 12.4
```
