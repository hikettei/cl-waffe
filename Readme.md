
# cl-waffe

Deeplearningのライブラリを自作しようとしている。。。冬休み終わったら時間なくて放棄しちゃうかも

# Todo/Memos


・Numclは使いたくないがOpenBLASに依存しないで動くようにしたい。

-> cl-waffe上ではcl標準のarrayで動作

-> numclベースの行列演算を書き換える

-> WaffeTensor >> (Numcl mgl-mat opencl-kernel...)


・reshape/randnなどの関数を自前実装(骨が折れる...)

-> 命名規則をどうにかする(logeとかwf-tanhとかはわかりずらい)

-> Tensorの型をちゃんと定義する + FP16/FP8

-> 次元配列の演算, ミニバッチ学習

-> (data tensor) = CL標準配列と定義

-> 学習データの読み込みがめちゃくちゃ遅い... MNISTですら重たい


・必要ライブラリの管理(versionなど)


・Error表示をちゃんと書く(invaild argument numberとかはわかりずらい)


・画像/テキストのloaderを作る CLのライブラリにいい感じのやつが見当たらない。。。


・ProgressBarに残り時間の推定を実装/Update-Frequencyを用意


・学習時のLossの推移をSLIME上かtxtに保存できるように書いておく


・えげちぃWarningの解消


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