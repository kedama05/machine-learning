# Neural Network  

1. パーセプトロン  
  kerasの単純なモデルはSequentialモデル  
  Sequentialモデル: ニューラルネットワークの層を積み上げる。  
  ex)  
    ```  
    from keras.model import Sequential  
    model = Sequential()  
    model.add(Dense(12, input_dim=8, kernel_initializer='random_uniform'))  
    ```  
  random_uniform: -0.05から0.05の範囲でランダムに初期化  
  random_normal: 平均0、分散0.05の正規分布に基づいた初期化   
  zero: 全ての重みを0で初期化  

2. 多重パーセプトロン  
  多重パーセプトロン: 多数の層を持つネットワーク  
  ニューラルネットワークは全結合層であり、層内の各ニューロンは、前の層にある全てのニューロンと次の層の全てのニューロンに接続されていることを意味する。  
  パーセプトロンは0か1の変化。  
  → 漸次学習はできない。  
  → 漸次的に変化するシグモイド関数を使用する。  

3. 活性化関数  
  シグモイド関数:  
    ```math  
    \sigma(x) = \frac {1} {1+exp(-x)}  
    ```  
  ReLU関数:  
    ```math  
    f(x)=max(0,x)  
    ```  

4. 勾配降下法  

5. 確率的勾配降下法  

