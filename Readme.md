
# cl-waffe

Deeplearningのライブラリを自作しようとしている。。。冬休み終わったら時間なくて放棄しちゃうかも

# Documents

Coming soon...


# Run MNIST

```
$ ./run-test-model.ros
```

please remain that it requiers roswell.


チュートリアルやサンプルコード、Documentなどはもう少ししたら書きます。

# Todo/Memos

・Write readme in english

・最適化とメモリ消費量を減らすのを頑張る

・Trainer使いやすくする

・Wikiを書く

・画像/テキストのloaderを作る CLのライブラリにいい感じのやつが見当たらない。。。


・ProgressBarに残り時間の推定を実装/Update-Frequencyを用意


・学習時のLossの推移をSLIME上かtxtに保存できるように書いておく


・モデルの計算式書く時にS式はしんどい、yaccで中置記法->S式に変換するマクロ作る (exa: (with-waffe-exp 1+ a))


・OptimizerにAdam/Momentum/RMSPropを実装 (Amos試してみたいな...)


・Embeddingやsinとか、機能を増やす LSTMの実装を目標に


プロファイリングして最適化を頑張る。。。


# Tutorials

numclは大きめのヒープを要求するので、roswellを使ってるなら以下のコマンドを入力した方がいい

```
$ ros config set dynamic-space-size 4gb
```

# Author

RuliaChan (Twitter: @ichndm)

### memos

; nodeのパラメーターの初期値にnil使えないのを覚えておく


## Environment

```
SBCL 2.2.5
macOS Monterey version 12.4
```


# Welcome To cl-waffe!

cl-waffeはCommonLispで実装されたDeeplearningのライブラリです。

cl-waffeでは以下の3つのマクロと2つの拡張用のマクロを用いてネットワークを定義していきます。
