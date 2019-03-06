# Neural Network  

## パーセプトロン  
kerasの単純なモデルはSequentialモデル  
Sequentialモデル: ニューラルネットワークの層を積み上げる。    
```Python
from keras.model import Sequential  
model = Sequential()  
model.add(Dense(12, input_dim=8, kernel_initializer='random_uniform'))  
```
random_uniform: -0.05から0.05の範囲でランダムに初期化  
random_normal: 平均0、分散0.05の正規分布に基づいた初期化  
zero: 全ての重みを0で初期化  


## 多重パーセプトロン  
多重パーセプトロン: 多数の層を持つネットワーク  
ニューラルネットワークは全結合層であり、層内の各ニューロンは、
前の層にある全てのニューロンと次の層の全てのニューロンに接続されていることを意味する。  
パーセプトロンは0か1の変化。  
→ 漸次学習はできない。  
→ 漸次的に変化するシグモイド関数を使用する。  


## 活性化関数  
シグモイド関数:  
入力(-∞,∞)が与えられたとき、(0,1)の範囲で変化する出力を得る。  
数学的にこの関数は連続。  
```math  
\sigma(x) = \frac {1} {1+exp(-x)}  
```  
ReLU関数:  
正規化線形関数  
非線形関数として表す。  
```math  
f(x)=max(0,x)  
```  
以降で、緩やかな変化が、学習アルゴリズムの開発において重要な要素であることを確認する。  
これらの学習アルゴリズムでは、重みを徐々に変化させ、ネットワークの出力誤差を減らすことが求められる。  
[Keras 活性化関数](https://keras.io/ja/activations/)

### 手書き数字認識  
MNIST  
>手書き数字のデータセット  
>6万件の学習データと1万件の評価データ  
>学習データには人手で正解が付与  
>グレースケールで28x28のピクセルで構成される


機械学習では、正解データが付与されたデータセットを利用できる場合、教師あり学習の一形態を適用できる。  
学習データセット → ネットワークをチューニング  
評価データセットをネットワークの学習に使用してはいけない。  

one-hotエンコーディング  
>カテゴリはd番目だけ1でそれ以外がすべて0  
>カテゴリ数分の次元の2進数ベクトルとして表現できる  
>学習アルゴリズムが数値を扱うことに特化しているデータマイニングの分野で非常に一般的な手法  

モデルを定義したあと、モデルを実行するためにコンパイルする必要がある。  
コンパイルする際にはいくつかのオプションを与えることができる。  
>モデルの学習時の重み更新に使用する最適化アルゴリズムを指定  
>損失関数も指定  
>学習したモデルを評価  

[Keras 損失関数](https://keras.io/ja/losses/)  

MSE  
>予測した値と真の値の平均二乗誤差  
>n個の予測ベクトルと\betaとn個の観測ベクトルYがあると仮定  
```math  
MSE=\frac{1}{n} \sum_{n}^{i=1}(\beta-Y)^2
```  

バイナリクロスエントロピー  
>モデルが予測した値p  
>真の値t  
```math
-t\log{p}-(1-t)\log{(1-p)}
```  

カテゴリカルクロスエントロピー  
>複数クラスの対数損失を計算する  
>モデルが予測した値`p_{i,j}`  
>真の値`t_{i,j}`  
```math
L_i=-\sum_{j}t_{i,j}\log{(p_{i,j})}
```  
[Keras 評価関数](https://keras.io/ja/metrics/)  

epoch  
>モデルが学習データセットに対して学習する回数  
>学習ごとに最適化アルゴリズムは損失関数の値を最小にするように重みを調整する  

bach-size  
>重みを更新する際に、データをいくつ使用するか  

callback  
>学習中の処理を変更  
>学習結果の保存  

検証データ  
>学習しながら学習状況の評価を行える  
>学習データの一部を検証データとして用いる  



## 勾配降下法  



## 確率的勾配降下法  


### script  
#### sample
keras\_MINST\_V1.py  
>dense layer: 1  
>activation: [softmax]  
>optimizer: SGD  
>NB\_EPOCH: 200  
>>Test score: 0.27738585779070857  
>>Test accuracy: 0.9227  

keras\_MNIST\_V2.py  
>dense layer: 3  
>activation: [relu, relu, softmax]  
>optimizer: SGD  
>NB\_EPOCH: 20  
>>Test score: 0.18603960166722536  
>>Test accuracy: 0.9462  

keras\_MNIST\_V3.py  
>dense layer: 3  
>activation: [relu, relu, softmax]  
>optimizer: SGD  
>NB\_EPOCH: 20  
>DROPOUT: 0.3  
>>Test score: 0.19942258299961685  
>>Test accuracy: 0.94  

keras\_MNIST\_V4.py  
>dense layer: 3  
>activation: [relu, relu, softmax]  
>optimizer: RMSprop  
>NB\_EPOCH: 20  
>DROPOUT: 0.3  
>>Test score: 0.10002544444922278  
>>Test accuracy: 0.9785  


#### original  
##### sgd  
----  
sgd.py  
>dense layer: 3  
>activation: [relu, relu, softmax]  
>optimizer: SGD  
>>Test score: 0.1860125882193446  
>>Test accuracy: 0.9463  

drop\_sgd.py  
>dense layer: 3  
>activation: [relu, relu, softmax]  
>optimizer: SGD  
>DROPOUT: 0.3  
>>Test score: 0.19941077888831496  
>>Test accuracy: 0.9401  

##### rmsprop  
----  
rmsprop.py  
>optimizer: RMSprop  
>DROPOUT: 0.3  
>>Test score: 0.13478955727134304  
>>Test accuracy: 0.9763  

drop\_rmsprop.py  
>optimizer: RMSprop  
>DROPOUT: 0.3  
>>Test score: 0.09949910521613092  
>>Test accuracy: 0.9785  

##### adam  
----  
adam.py  
>optimizer: Adam  
>DROPOUT: 0.3  
>>Test score: 0.11095440730602135  
>>Test accuracy: 0.9776  

drop\_adam.py  
>optimizer: Adam  
>DROPOUT: 0.3  
>>Test score: 0.07498836264909478  
>>Test accuracy: 0.9794  

##### adamax  
----
adamax.py  
>optimizer: Adamax  
>DROPOUT: 0.3  
>>Test score: 0.07737895044728939  
>>Test accuracy: 0.98  

drop\_adamax.py  
>optimizer: Adamax  
>DROPOUT: 0.3  
>>Test score: 0.07470145506088738  
>>Test accuracy: 0.977  

##### Nadam  
----  
nadam.py  
>optimizer: Nadam  
>DROPOUT: 0.3  
>>Test score: 0.13653561498492076  
>>Test accuracy: 0.976  

drop\_nadam.py  
>optimizer: Nadam  
>DROPOUT: 0.3  
>>Test score: 0.07784218108507267  
>>Test accuracy: 0.978  


