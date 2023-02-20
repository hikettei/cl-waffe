
(in-package :cl-waffe-test)

(in-suite :test)

(defmodel TestModel1 nil
  :parameters nil
  :forward ((x)
	    (!exp x)))


(defmodel TestModel2 nil
  :parameters ((x nil))
  :forward ((x)
	    (save-for-backward x x)
	    (!exp x)))

(defnode TestNodebackward nil
  :parameters ((x nil))
  :forward ((x)
	    (save-for-backward x x)
	    (!exp x))
  :backward ((dy) (list dy)))

(defnode TestNode1 nil
  :parameters ((layer (TestNodeBackward))
	       (x nil))
  :forward ((x)
	    (save-for-backward x x)
	    (call (self layer) x))
  :backward ((dy) (list dy)))

; testing side effects never apparented.
(defun test-model1 ()
  (let ((a (!randn `(10 10))))
    (not (M= (data a) (value (call (TestModel1) a))))))

; copy is enabled?
(defun test-save-for-backward ()
  (let ((m (TestModel2))
	(p (parameter (!randn `(10 10)))))
    (call m p)
    (slot-value m 'x)))

; copy is ok?
(defun test-save-for-backward1 ()
  (let ((m (TestModel2))
	(p (!randn `(10 10))))
    (call m p)
    (null (slot-value m 'x))))

; copy is disabled?
(defun test-save-for-backward2 ()
  (let ((m (TestModel2))
	(p (!randn `(10 10))))
    (with-no-grad
	(call m p)
      (null (slot-value m 'x)))))

; deeper node won't do copy
(defun test-save-for-backward3 ()
  (let ((m (TestNode1))
	(p (parameter (!randn `(10 10)))))
    (call m p)
    (and (slot-value m 'x)
	 (null (TestNodeBackward-x (TestNode1-layer m))))))

; this is the same to const
(defun test-save-for-backward4 ()
  (let ((m (TestNode1))
	(p (!randn `(10 10))))
    (call m p)
    (not (and (slot-value m 'x)
	      (null (TestNodeBackward-x (TestNode1-layer m)))))))

(defnode TestNodeExp1 nil
  :parameters ((layer (TestNode1))
	       (x nil))
  :forward ((x)
	    (save-for-backward x x)
	    (call (self layer) x))
  :backward ((dy) (list dy)))

(defnode TestNodeExp nil
  :parameters ((layer (TestNodeExp1))
	       (x nil))
  :forward ((x)
	    (save-for-backward x x)
	    (call (TestNodeExp1) x))
  :backward ((dy) (list dy)))

(defmodel TestNodeCallerModel nil
  :parameters ((layer (TestNodeExp))
	       (x nil))
  :forward ((x)
	    (save-for-backward x x)
	    (call (self layer) x)))

; for complicated model (Node -> Node -> Node -> Node)
(defun test-save-for-backward5 ()
  (let ((a (parameter (!randn `(10 10))))
	(model (TestNodeCallerModel)))
    (call model a)
    (and (slot-value model 'x)
	 (slot-value (slot-value model 'layer) 'x)
	 (null (slot-value
		(slot-value
		 (slot-value model 'layer)
		 'layer)
		'x))
	 (null
	  (slot-value
	   (slot-value
	    (slot-value
	     (slot-value model 'layer)
	     'layer)
	    'layer)
	   'x)))))

#|
Node内部のsave-for-backwardは以下の条件で無視される

(with-no-grad)の内部 (逆伝播が呼び出されることはないから) :backwardスロットの中で!mul等標準の命令を使っても遅くならない。

is-ancestor-param=nilであるTensor (that is 逆伝播時に辿ることのない経路にあるTensor)

defnode内部のdefnode
-> 例えば:forwardスロットの中で(!exp)を使いたい この時モデルを逆伝播すると一番上のDefnodeの:backwardを辿りそれ以下の:backwardは呼び出されないから Save-for-backwardする必要はない。
|#

(test save-for-backward-test
      (is (test-model1))
      (is (test-save-for-backward))
      (is (test-save-for-backward1))
      (is (test-save-for-backward2))
      (is (test-save-for-backward3))
      (is (test-save-for-backward4))
      (is (test-save-for-backward5)))
