
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

numclのバージョン管理

モデルの表示(like torch keras)をしたいから、defmodelを二種類作る

Kernelの命令のwiki作る

termplotでlossのグラフを表示する

Error表示わかりやすくする. (invaild argument numberとかどの関数？ってなる)

DLWaffeみたいなTrainer Classを実装

画像, テキスト読み込み用のライブラリを作る

Seq2Seqを実装

Transformerを実装

# Kernel Instructions

:add ... 3DArray + 3DArray

# 行列演算と微分を増やすin numcl
# Optimizer Model