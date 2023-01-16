
# English version is coming soon... i promise that i will rewrite this draft when i am free

勉強と趣味で書いてるんで、実用的になるかわからんけど・・・

チュートリアルを書くついでに、cl-waffeの設計とか、TODOをここに書いておきます。

まだ僕にわかなので、間違ってる箇所があったらすみません。

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

cl-waffeはCommonLispで実装された深層学習のためのライブラリです。以下を目標に設計しました。

・簡潔に数式の記述が可能、学習が簡単

・CPU/GPU両計算時で高速に動作する

・Define by run (計算と同時にノードを構築)

cl-waffeを利用してモデルの設計を始めるにあたって、最低でも(#サンプルコードとデータローダー)という項までは目を通して頂けるとすぐ使い始められると思います。

## モデルと計算ノード

個人的に昔からChainerの設計が好きで、cl-waffeの設計もそれに強く影響を受けています。

cl-waffeでは自動微分をサポートしていますから、他の同様のライブラリのように実行時に計算ノードを生成します。

### Define by run vs Define and run

背景として、Deeplearningのライブラリには、`define by run`という設計思想と、それに対称的な`define and run`という設計思想が存在します。

架空のライブラリを用いて、軽い実例を提示します。

(Tensorflowあまり扱ってこなかったので、間違ってたら申し訳ない・・・)

### 1. Define by run

```
a = tensor(1)
b = tensor(2)

if ~~~:
    z = a * b
else:
    z = a + b ; 実行時に辿ったルートを基に計算ノードを作成、仮にEpochごとに辿るルートが変わってもノードが生成される。
```

### 2. Define and run

```
a = tensor(1)
b = tensor(2)

if ~~~:
    z = a * b
else:
    z = a + b ; コードが定義された時のルートを基に計算ルートを作成、最適化がしやすいが柔軟性が失われる欠点がある。
```

cl-waffeでは、1.の`Define by run`で設計しています。計算ノードの最適化が難しいという欠点がありますが、CommonLispの柔軟性と後述(#非破壊的代入と破壊的代入)の`(with-waffe-expression)`というマクロを用いて計算ノードを最適化して、この問題を解決するつもりです。**(This is included in TODO 多分きっとそうなる予定)**


### モデルとノードを定義する

cl-waffeでは、ノードを定義するために`defmodel` `defnode`という二つのマクロを提供しています。

それではまず、`z = (!mul a b)`という計算を、逆伝搬に対応しつつ行う場合から考えていきましょう。

cl-waffe!mulの定義は以下のようになっています。(簡単のため一部省略)


```lisp
(defun !mul (x y)
  (call (MulTensor) x y))
```


関数`!mul`が呼び出された時、マクロ`defnode`(使い方は後述)で定義された構造体`MulTensor`を初期化し、関数`call`を用いて、構造体`MulTensor`に対応したメソッド`call-forward`を呼び出しています。

そのため、ユーザーは、計算が行われるたびに生成される構造体(この場合`MulTensor`)の雛形を定義すれば、逆電波に対応したネットワークが作成できます。

そのためのマクロ`cl-waffe:defnode`を実例と一緒に紹介します。


**マクロ`(defnode name args &key parameters forward backward optimize)`**

**Return: nil**

  構造体`(gensym (symbol-name name))`を定義します。(学習時大量に生成するので内部的にはただの構造体です)

  構造体のスロットは、`parameters`に応じて定義されます。


**parameters -> (変数名 初期値)若しくは(変数名 初期値 :type 型)で構成されるリスト**

  マクロ`defnode`を評価すると、式`(defun name (&rest init-args) (構造体の生成))`が展開されます。

  ですから、シンボル`name`というノードを初期化するときは、式`(name args1 args2 ...)`のように呼び出します。

  構造体は:constructorを持ちますから、リスト`args`に応じた引数を上記の式に入力することで、Parameterの初期値を変更できます。


**forward -> list, (引数 body)**

 この関数の返り値の型は`WaffeTensor`である必要があります。

引数に指定されたTensorは、cl-waffeの自動微分によって勾配が求められます。（ただしこれは、任意の条件によって最適化され除外される場合もある）


**backward -> list, (引数 body)**

 この関数の返り値は`list`である必要があります。（後述）


forward, backwardスロットに定義された関数は一度`defnode`と同じスコープに関数`(gensym)`として定義されます。


内部ではこれらを呼び出すために、総称関数`(call-backward model)`, `(call-forward model)`を呼び出していますが、通常の利用であれば`(call)`でforward関数を呼び出せます。


`defnode`で定義された計算ノードは、自動微分の範囲外であるため、必ず:forward :backwardスロットが必要である点に留意してください。

**optimize -> boolean**

  :forward :backward関数内で`(declare (speed 3) (space 0) (safety 0))`をつけるかどうか？(default: nil)

### Backward関数の記述の仕方

cl-waffeの自動微分はトップダウン型です。

例えばforward関数の引数が`(x y)`であったとします。

backward関数の引数`(dy)`(これは固定)は、計算グラフで言えば一つ上のノードの勾配を受け取ります。

backward関数はこれをもとに、`(list 次のノードのxの勾配 次のノードのyの勾配)`を返してください。

Example:

```lisp
; operators.lispから抜粋

(defnode MulTensor nil
  :optimize t
  :parameters ((xi T) (yi T))
  :forward ((x y)
      (setf (self xi) (data x))
      (setf (self yi) (data y))
      (with-searching-calc-node :mul x y))
  :backward ((dy) (list (!modify (self yi) :*= dy)
      (!modify (self xi) :*= dy))))

;Example: (call (MulTensor) tensor1 tensor2)

(defnode MeanTensor (axis)
  :optimize t
  :parameters ((axis axis) (repeats T))
  :forward ((x)
      (setf (self repeats) (assure-tensor (!shape x (self axis))))
      (with-searching-calc-node :mean x (self axis)))
  :backward ((dy) (list (!repeats dy (self axis) (self repeats)))))

;Example: (call (MeanTensor 1) tensor1)
```

### Forward/Backwardにおいて、Parameterと値のやり取りをする。

forward/backward関数の内部において、macroletで定義されたマクロ`(self name)`があります。

これを用いて、Parameterと値をやり取りします。

### モデルを定義する

前の項で紹介した`defnode`は自動微分に対応しておらず、内部の処理を定義するために作られたマクロなので、通常ユーザーが使う必要はありません。

`defnode`で構成された計算ノードを基にDeeplearningモデルを定義するためには、マクロ`defmodel`を使います。

**マクロ`(defmodel name args &key parameters forward backward hide-from-tree optimize)`**

`defnode`と基本同じですが、`backward`スロットは基本的にnilを指定し、自動微分で勾配を求めます。

**`hide-from-tree` -> boolean, Tの時`defnode`と同義として扱われる。`(default -> nil)`**

この`defmodel`マクロを使って、簡単なMLPモデルを実装してみましょう。

cl-waffeでは、パッケージ`cl-waffe.nn`にLinearやCNNなど、基本的なモデルが標準で定義されています。

```lisp
(defmodel MLP (activation)
  :parameters ((layer1   (denselayer (* 28 28) 512 T activation))
               (layer2   (denselayer 512 256 T activation))
               (layer3   (linearlayer 256 10 T)))
  :forward ((x)
      (call (self layer3)
            (call (self layer2)
                  (call (self layer1))))
```

このように定義されますが、:forwardのネストが深くて不恰好です。

それを解決するためにマクロ`(with-calling-layers input &rest layers)`がexportされています。

速度の面からも基本的に`(with-calling-layers)`を使う方が優れています、詳細は次の項で解説されています。

このように書き換えることができます。

```lisp
:forward (x)
  (with-calling-layers x
    (layer1 x)
    (layer2 x)
    (layer3 x))

    ; => x
```

補足ですが、x以外の引数を取りたい場合は、


```lisp
:forward (x)
  (with-calling-layers x
    (layerA x 1 2)
    (layerB x a)
    (layerC x))

    ; => x
```

のように記述できます。

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

この問題を解決するため、cl-waffeではユーザーが計算ノードを生成しながら高速に記述するために、以下のマクロが提供されています。

**function `(!modify target instruction &rest args)`**

**Return: 破壊された`target`**

  `target`を破壊的に代入して、`instruction`に応じたカーネルを`args`を引数として呼び出します。

  **この関数は計算ノードは生成しません**, そのため、:forwardスロットの内部や、`defnode`など、自動微分を利用しない関数の:backwardスロットで利用できますが、後述の`with-waffe-expression`が一番簡潔です。

Example:

```
Coming Soon...
```


**macro `(with-calling-layers input &rest layers)`**

**Returns: 計算されたinput**

`defmodel` `defoptimizer` `defnode` 内部において、inputを引数として、inputを破壊的に代入しながらlayersを適用していきます。

layerが適用される順番は、上から下です。

Example:

```
Coming soon...
```

**macro `(with-waffe-expression ... )` <= Todo**

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
