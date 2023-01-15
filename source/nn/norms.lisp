
(in-package :cl-waffe.nn)

(defmodel BatchNorm2d (in-features &key (affine t) (epsilon 1e-7))
  :optimize nil
  :parameters ((affine (if affine
			   (linearlayer in-features in-features T)
			   T)
		       :type linearlayer)
	       (epsilon epsilon :type float))
  :forward ((x)
	    (let* ((average (!mean x 0 t))
		   (var (!mean (!sub x average)))
		   (r (!div (!sub average var)
			    (!sqrt (!add (!pow var 2) (self epsilon))))))
	      (if (eql (self affine) T)
		  r
		  (call (self affine) r)))))
