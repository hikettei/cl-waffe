
mkdir tmp
cd ./tmp
curl -O https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.bz2
curl -O https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.t.bz2
bzip2 -d ./mnist.scale.bz2
bzip2 -d ./mnist.scale.t.bz2
cd ../
