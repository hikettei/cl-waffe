
(in-package :cl-waffe.io)

(defmacro do-index-value-list ((index value list) &body body)
  (let ((iter (gensym))
        (inner-list (gensym)))
    `(labels ((,iter (,inner-list)
                     (when ,inner-list
                       (let ((,index (car ,inner-list))
                             (,value (cadr ,inner-list)))
                         ,@body)
                       (,iter (cddr ,inner-list)))))
       (,iter ,list))))

(defun read-libsvm-data (data-path data-dimension n-class &key (most-min-class 1))
  ; Todo: This is temporary and rewrite it for cl-waffe.
  ; This API will be change in the near future.
  (let* ((data-list (svmformat:parse-file data-path))
         (len (length data-list))
         (target     (make-array (list len n-class)
				       :element-type 'float
				       :initial-element 0.0))
         (datamatrix (make-array (list len data-dimension)
				       :element-type 'float
				       :initial-element 0.0)))
    (loop for i fixnum from 0
          for datum in data-list
          do (setf (aref target i (- (car datum) most-min-class)) 1.0)
             (do-index-value-list (j v (cdr datum))
               (setf (aref datamatrix i (- j most-min-class)) v)))
    (values (const (mgl-mat:array-to-mat datamatrix))
	    (const (mgl-mat:array-to-mat target)))))
