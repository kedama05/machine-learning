# Neural Network  

## パーセプトロン  
kerasの単純なモデルはSequentialモデル  
Sequentialモデル: ニューラルネットワークの層を積み上げる。  
ex)
    from keras.model import Sequential  
    model = Sequential()  
    model.add(Dense(12, input_dim=8, kernel_initializer='random_uniform'))  

random_uniform: -0.05から0.05の範囲でランダムに初期化  
random_normal: 平均0、分散0.05の正規分布に基づいた初期化   
zero: 全ての重みを0で初期化  



## 多重パーセプトロン  


## 活性化関数  


## 勾配降下法  


## 確率的勾配降下法  



