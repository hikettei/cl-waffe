
# cl-waffe

Deeplearningのライブラリを自作しようとしている。。。冬休み終わったら時間なくて放棄しちゃうかも

チュートリアルやサンプルコード、Documentなどはもう少ししたら書きます。

# Todo/Memos


・ 次元配列の演算, ミニバッチ学習


・ 学習データの読み込みがめちゃくちゃ遅い... MNISTですら重たい

-> Dataloaderのベンチマーク


・必要ライブラリの管理(versionなど)


・画像/テキストのloaderを作る CLのライブラリにいい感じのやつが見当たらない。。。


・ProgressBarに残り時間の推定を実装/Update-Frequencyを用意


・学習時のLossの推移をSLIME上かtxtに保存できるように書いておく


・モデルの計算式書く時にS式はしんどい、yaccで中置記法->S式に変換するマクロ作る (exa: (with-waffe-exp 1+ a))


・BackendをCPU/Cuda(OpenCl)に増やす


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