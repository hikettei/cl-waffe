
# English version is coming soon... i promise that i will rewrite this draft in english when i am free.

勉強と趣味で書いてるんで、実用的になるかわからんけど・・・

自分まだにわかなんで、間違ってるところあったら許して欲しい〜〜

チュートリアルを書くついでに、cl-waffeの設計とか、TODOをここに書いておきます。

将来こんな感じにしたいな〜〜〜ってやつ、所詮趣味やから何年かかるかわからんけど

半分妄想です、未実装の機能も含まれている点にご留意ください。

TODOリストだと思ってみてくれたら

# cl-waffe tutorials

- [導入](#導入)
  - [モデルと計算ノード](#モデルと計算ノード)
  - [非破壊的代入と破壊的代入](#非破壊的代入と破壊的代入)
  - [サンプルコードとデータローダー](#サンプルコードとデータローダー)
  - [テンソルの使い方](#テンソルの使い方)
  - [ライブラリを拡張する](#ライブラリを拡張する)
  - [Trainer](#Trainer)
  - [モデルを保存/復元する](#モデルを保存/復元する)
  - [CUDAを使って学習する](#CUDAを使って学習する)
  - [References](#References)

## TODO

Warning/Errorの表示をちゃんとする

もっと高速化

(with-waffe-expression)マクロを用いて、Forward関数の自動最適化

DataLoaderのライブラリを作る

Trainer関数を書き直す

LossをグラフにPlotする

それらが終わったら、CNNやRNNなど基本的なモデルを実装していく・・・

良いTestコードを書く

ちゃんとした条件でBenchmarkを計測

## 導入

cl-waffeはCommonLispの深層学習ライブラリです。以下を目標に設計しました。

・簡潔に数式の記述が可能、学習が簡単

・CPU/GPU両計算時で高速に動作する

・Define by run (計算と同時にノードを構築)

cl-waffeを利用してモデルの設計を始めるにあたって、最低でも以下の三つの項目には目を通して頂けるとすぐ使い始められると思います。

## モデルと計算ノード

### ライブラリの構造

自分は元々Chainerの設計が好きだったので、cl-waffeの設計もそれに強く影響されています。

`Define by run`という設計思想のDeeplearningライブラリは、`Define and run`という設計思想と対称的に、実行と同時に計算ノードを生成します。

架空のライブラリで実例を提示します。

### 1. Define by run

```
x = tensor(1)
y = tensor(2)

if ~~~:
    a = x * y
else:
    a = x + y ; 実行時に辿ったルートを基に計算ノードを生成する。
```

### 2. Define and run 

(ChainerとTorchしか触ってこなかった人間なので間違ってるかも)

```
x = tensor(1)
y = tensor(2)

if ~~~:
    a = x * y
else:
    a = x + y ; 実行と同時に計算ノードが定義されるので、処理によって分岐が変わることは許されない
```

cl-waffeは1のdefine by runで設計されています。

define by runは計算ノードの最適化が難しいという欠点がありますが、次の項で述べる`(with-waffe-expression)`マクロを用いてそれを解決しています。(TODO, 多分将来的にそうなる予定)

### ノードの定義

cl-waffeでは、毎回計算をする時に生成される計算ノードを、場合に応じて`defmodel`, `defnode`, `defoptimizer`というマクロを用いて定義します。

文章量の都合上、`defnode`と`defoptimizer`は発展的なので、(#ライブラリを拡張する)の項目で触れています。


## 非破壊的代入と破壊的代入

コンピューターで数値計算（特に行列など）を扱うとき、メモリの確保の手法に以下の2通りがあります。


・破壊的代入（引数を保存していたメモリ領域に計算の結果を上書きする）

・非破壊的代入（引数を保存していたメモリ領域とは別に、結果を保存する領域を確保し、計算の結果を記録する）


破壊的代入は、計算を記述するときに、式が煩雑になりやすいというデメリットがありますが、計算のたびに新しい領域を確保する必要がないので、非常に高速です。

（自動微分の時に使用する入力の値は保存しないといけないので、そこは上書きしてはいけない）

非破壊的代入は、計算の記述が直感的ですが、計算のたびに新しい領域を確保する必要があり、後述の結果から自分の環境では破壊的代入の約4倍ほど遅いです。

実際に、自分の環境(SBCL, OpenBLAS)で計測した結果がこちらです。

a, bは!randnで生成された、1024x1024の乱数列で、小数点は:floatで計算しています。

cl-waffeで600回, a+bを計算してみます。

```lisp
(time (dotimes (i 600) (!add a b))) ; 非破壊的に代入

;=>
Evaluation took:
  0.808 seconds of real time
  1.014707 seconds of total run time (0.866665 user, 0.148042 system)
  [ Run times consist of 0.048 seconds GC time, and 0.967 seconds non-GC time. ]
  125.62% CPU
  124 lambdas converted
  1,863,645,748 processor cycles
  7 page faults
  2,524,014,528 bytes consed
  

(with-no-grad ; 計算ノードを生成せずに計算
  (time (dotimes (i step-num) (!add a b))))
  
;=>
Evaluation took:
  0.758 seconds of real time
  0.986515 seconds of total run time (0.860935 user, 0.125580 system)
  [ Run times consist of 0.078 seconds GC time, and 0.909 seconds non-GC time. ]
  130.21% CPU
  1,747,819,084 processor cycles
  2,518,120,288 bytes consed


(time (dotimes (i step-num) (!modify a :+= b)) ;破壊的に代入
;=>
Evaluation took: 
  0.153 seconds of real time
  0.432072 seconds of total run time (0.402585 user, 0.029487 system)
  282.35% CPU
  51 lambdas converted
  352,627,488 processor cycles
  2,985,584 bytes conses
  
(time (dotimes (i step-num) (mgl-mat:axpy! 1.0 (data a) (data b)))) ; 番外編 mgl-matを直接呼び出す (毎回値が変わってるねこれ...)

Evaluation took:
  0.142 seconds of real time
  0.419678 seconds of total run time (0.392220 user, 0.027458 system)
  295.77% CPU
  327,691,366 processor cycles
  490,848 bytes consed
```

この場合、破壊的代入は非破壊的代入の約5.28倍早いという結果になりました。

これは、計算のbackendに使用しているmgl-matという行列演算のライブラリが、破壊的代入を前提として設計されているからです。

そのため、ユーザーが計算ノードを生成しながら高速に記述するために、以下のマクロが提供されています。

`(!modify target instruction &rest args)`

`target`を破壊的に代入して、`instruction`に応じたカーネルを`args`を引数として呼び出します。

**この関数は計算ノードは生成しません**, そのため、:forwardスロットの内部や、`defnode`など、自動微分を利用しない関数の:backwardスロットで利用できますが、後述の`with-waffe-expression`が一番簡潔です。

Example:

```
Coming Soon...
```


`(with-calling-layers input &rest layers)`

Returns: 計算されたinput
`defmodel` `defoptimizer` `defnode` 内部において、inputを引数として、inputを破壊的に代入しながらlayersを適用していきます。

layerが適用される順番は、上から下です。

Example:

```
Coming soon...
```

`(with-waffe-expression ... )` <= Todo

`with-waffe-expression`内部では、中置記法をサポートしているので、(個人的にはS式が好きなんですが・・・)、見慣れた手順でForwardProcessを定義できます。

Example:

```
Coming soon...
```

## サンプルコードとデータローダー

## テンソルの使い方

## 標準実装の機能

## ライブラリを拡張する

## Trainer

## モデルを保存/復元する

## CUDAを使って学習する

## References
