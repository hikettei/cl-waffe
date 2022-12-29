
# cl-waffe


# 目標

自動微分 PyTorchのようなわかりやすい設計

CPU OpenCL Cudaなどのbackend

行列演算はnumclベースにする。色々な環境で動くようにする。

最適化を頑張る。。。

FP16 FP8 FP4

Dataset Model Optimizerの統合、使いやすくする

Trainer Classに学習する時のlambda式を渡すみたいな。。。

# Tutorials

numclは大きめのヒープを要求するので、roswellを使ってるなら以下のコマンドを入力した方がいい

```
$ ros config set dynamic-space-size 4gb
```

# Author

RuliaChan (Twitter: @ichndm)

### Todo

defmodelでParameterの初期値をnilにすると代入できなくなる、後から代入したいならTを使うように

行列演算
numclでCPUのカーネル
openclでopenclのカーネル
それぞれのカーネルようにデータ型を定義する
numclのバージョン管理