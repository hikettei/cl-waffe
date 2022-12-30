
(in-package :cl-waffe.optimizers)

; modelからParameterの検索をする

(defmacro is-waffe-model (model)
  ; Parameterの値に、defmodelを介して定義したわけではないが以下のslotを持つ構造体があると積む
  `(and (slot-exists-p ,model 'parameters)
        (slot-exists-p ,model 'hide-from-tree)
        (slot-exists-p ,model 'forward)
        (slot-exists-p ,model 'backward)))


(defun find-parameters ())

