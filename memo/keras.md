keras  
====  
# BackBone  
## VGG16  
## VGG19  
## Resnet 50  

## inceptionV3  
## xception  

# Function  
## fit  
固定回数（データセットの反復）の試行でモデルを学習させます．  

### callback  
トレーニング手順の特定の段階で適用される一連の関数  

Callback  
BaseLogger  
TerminateOnNaN  
ProgbarLogger  
History  
ModelCheckpoint  
EarlyStopping  



# Class  
## ImageDataGenerator  
    リアルタイムにデータ拡張しながら，テンソル画像データのバッチを生成します．また，このジェネレータは，データを無限にループするので，無限にバッチを生成します．
    from keras.preprocessing.image import ImageDataGenerator
        ImageDataGenerator(
            featurewise_center=False,
                # 真理値．データセット全体で，入力の平均を0にします．
            samplewise_center=False,
                # 真理値．各サンプルの平均を0にします．
            featurewise_std_normalization=False,
                # 真理値．データセット全体で，入力の平均を0にします．
            samplewise_std_normalization=False,
                # 真理値．各入力をその標準偏差で正規化します．
            zca_whitening=False,
                # 真理値．ZCA白色化を適用します．
            zca_epsilon=1e-06,
                # ZCA白色化のイプシロン．デフォルトは1e-6．
            rotation_range=0.0,
                # 整数．画像をランダムに回転する回転範囲．
            width_shift_range=0.0,
                # 浮動小数点数（横幅に対する割合）．ランダムに水平シフトする範囲．
            height_shift_range=0.0,
                # 浮動小数点数（縦幅に対する割合）．ランダムに垂直シフトする範囲．
            brightness_range=None,  # 
            shear_range=0.0,
                # 浮動小数点数．シアー強度（反時計回りのシアー角度）．
            zoom_range=0.0,
                # 浮動小数点数または[lower，upper]．ランダムにズームする範囲．浮動小数点数が与えられた場合，[lower, upper] = [1-zoom_range, 1+zoom_range]です．
            channel_shift_range=0.0,
                # 浮動小数点数．ランダムにチャンネルをシフトする範囲．
            fill_mode='nearest',
                #  {"constant", "nearest", "reflect", "wrap"}のいずれか．デフォルトは 'nearest'です．指定されたモードに応じて，入力画像の境界周りを埋めます．
            cval=0.0,
                # 浮動小数点数または整数．fill_mode = "constant"のときに境界周辺で利用される値．
            horizontal_flip=False,
                # 真理値．水平方向に入力をランダムに反転します．
            vertical_flip=False,
                # 真理値．垂直方向に入力をランダムに反転します．
            rescale=None,
                # 画素値のリスケーリング係数．デフォルトはNone．Noneか0ならば，適用しない．それ以外であれば，(他の変換を行う前に) 与えられた値をデータに積算する．
            preprocessing_function=None,
                # 各入力に適用される関数です．この関数は他の変更が行われる前に実行されます．この関数は3次元のNumpyテンソルを引数にとり，同じshapeのテンソルを出力するように定義する必要があります．
            data_format=None,
                # {"channels_first", "channels_last"}のどちらか．         
            validatIon_split=0.0
                # 浮動小数点数．検証のために予約しておく画像の割合（0から1）
        )


        
