
# English version is coming soon... i will rewrite it when i am free.

チュートリアルを書くついでに、cl-waffeの設計とか、TODOをここに書いておきます。

未実装の機能も含まれている点にご留意ください。

# cl-waffe tutorials

## 導入

cl-waffeはCommonLispの深層学習ライブラリです。以下を目標に設計しました。

・簡潔に数式の記述が可能、学習が簡単

・CPU/GPU両計算時で高速に動作する

・Define by run (計算と同時にノードを構築)

cl-waffeを利用してモデルの設計を始めるにあたって、最低でも以下の三つの項目には目を通して頂けるとすぐ使い始められると思います。

## モデルと計算ノード

## 非破壊的代入と破壊的代入 高速なコードを書く手法

## MNISTでの例, データローダーの使い方

## テンソルの使い方

## 標準実装の機能

## ライブラリを拡張する

## Trainer Class, Train関数の使い方

## モデルを保存/復元する

## CUDAを使って学習する