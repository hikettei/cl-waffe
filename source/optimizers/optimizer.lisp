
(in-package :cl-waffe.optimizers)

; modelからParameterの検索をする

(defmacro is-waffe-model (model)
  ; Parameterの値に、defmodelを介して定義したわけではないが以下のslotを持つ構造体があると積む
  `(and (slot-exists-p ,model 'cl-waffe:parameters)
        (slot-exists-p ,model 'cl-waffe:hide-from-tree)
        (slot-exists-p ,model 'cl-waffe:forward)
        (slot-exists-p ,model 'cl-waffe:backward)))

(defun find-parameters (model)
  (let ((parameters `(T)))
    (labels ((search-param (m)
	       (if (is-waffe-model m)
		   (dolist (p (slot-value m 'parameters))
		     (search-param (slot-value m p)))
		   (if (typep m 'cl-waffe:WaffeTensor)
		       (unless (null (slot-value m 'grad))
			   (push m parameters))))))
      (search-param model)
      (if (= (length parameters) 1)
	  (error "Could not find any parameter")
	  (butlast parameters)))))

