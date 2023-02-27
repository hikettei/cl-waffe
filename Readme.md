
# cl-waffe

![cl-waffe](https://hikettei.github.io/cl-waffe-docs/docs/cl-waffe-logo.png)

**This package is still under development, and its features are far from practical.**

cl-waffe is a deep learning framework with modern APIs for Common Lisp.

# Documents

[Documentation](https://hikettei.github.io/cl-waffe-docs) is available.

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

## Broadcasting Matrix.

See also: [Document](https://hikettei.github.io/cl-waffe-docs/docs/using-tensor.html#broadcasting)

```lisp
(setq a (!randn `(100 100 100)))
(setq b (!randn `(100 1 100)))

(time (!add a b))
;Evaluation took:
;  0.007 seconds of real time
;  0.006872 seconds of total run time (0.006766 user, 0.000106 system)
;  100.00% CPU
;  16,330,130 processor cycles
;  4,065,280 bytes consed
  
;#Const((((-1.65... 0.359... ~ -1.04... -1.63...)         
;                   ...
;         (0.172... 0.716... ~ -1.21... -0.18...))        
;                 ...
;        ((3.434... 0.050... ~ -0.65... 0.924...)         
;                   ...
;         (3.686... -1.60... ~ -0.31... 1.250...))) :mgl t :shape (100 100 100))
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
;#Const((((-1.47... 1.016... ~ -1.29... -1.71...)         
;                   ...
;         (2.276... 0.878... ~ -1.35... 0.466...))        
;                 ...
;        ((1.712... 1.318... ~ 0.213... 1.262...)         
;                   ...
;         (1.084... -0.18... ~ -1.42... 0.552...))) :mgl t :shape (100 100 100))

(time (!aref a 0 0 0))
;Evaluation took:
;  0.000 seconds of real time
;  0.000140 seconds of total run time (0.000113 user, 0.000027 system)
;  100.00% CPU
;  217,178 processor cycles
;  0 bytes consed
  
;#Const((((-1.47...))) :mgl t :shape (1 1 1))

(time (!aref a t 0 0))
;Evaluation took:
;  0.000 seconds of real time
;  0.000153 seconds of total run time (0.000132 user, 0.000021 system)
;  100.00% CPU
;  275,596 processor cycles
;  65,024 bytes consed
  
;#Const((((-1.47...))        
;                 ...
;        ((1.712...))) :mgl t :shape (100 1 1))

(time (!aref a '(0 3) '(10 -1) t))

;Evaluation took:
;  0.004 seconds of real time
;  0.004643 seconds of total run time (0.004562 user, 0.000081 system)
;  125.00% CPU
;  11,068,610 processor cycles
;  4,346,976 bytes consed
  
;#Const((((1.400... -0.64... ~ -0.48... 2.753...)         
;                   ...
;         (-0.07... 1.025... ~ 0.765... 0.371...))        
;                 ...
;        ((-0.47... 0.320... ~ 1.465... 1.738...)         
;                   ...
;         (1.566... -2.02... ~ -0.30... 0.085...))) :mgl t :shape (3 89 100))

(time (setf (!aref a '(0 3)) (!ones '(100 3))))
```

## Auto backpropagations

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

## Extensible APIs

See also: [Document](https://hikettei.github.io/cl-waffe-docs/docs/extend-library.html)

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

# Usage

For example:

```
$ git clone git@github.com:hikettei/cl-waffe.git
$ cd cl-waffe
$ sbcl
* (load "cl-waffe.asd")
* (ql:quickload :cl-waffe) ; all is done!
```

# Run MNIST With Roswell

```
$ cd examples
$ sh install.sh
$ cd ..
$ ./run-test-model.ros mnist
```

# Problems/Todo

・破壊的代入のサポート(Support more destructive operations)

・Neural Networkの追加 (Add cl-waffe.nn models)

・IterationのBackwardを高速化 (Improve performance of RNN)

・モデルの保存に対応 (Save and restore trained models.)

・グラフの表示に対応 (Plotting losses and so on)

・様々なデータ構造を扱えるように (Support more types of data structure)

・性能向上（メモリ使用量/CPU使用率の観点から
）(In term of cpu-usage rate/memory-usage, cl-waffe has a lot of challenge to performance.)

・CUDAに対応 (Support CUDA)

・他の処理系で動くか試す (Try on another systems (e.g.: CCL))

# Author

hikettei (Twitter: @ichndm, github:@hikettei)


## Environment

```
SBCL 2.2.5
macOS Monterey version 12.4
```
