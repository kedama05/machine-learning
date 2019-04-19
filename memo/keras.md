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

#### Callback  
    keras.callbacks.Callback()  
新しいコールバックを構築する  
#### BaseLogger  
    keras.callbacks.BaseLogger()  
監視されている評価値のエポック平均を蓄積する  
#### TerminateOnNaN  
    keras.callbacks.TerminateOnNaN()  
損失がNaNになった時に訓練を終了する  
#### ProgbarLogger  
    keras.callbacks.ProgbarLogger(count_mode='samples')  
標準出力に評価値を出力する
#### History  
    keras.callbacks.History()  
Historyオブジェクトにイベントを記録する  
Historyオブジェクトはモデルのfitメソッドの返り値として取得する  
#### ModelCheckpoint  
    keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
各エポック終了後にモデルを保存する  

    filepath: 文字列，モデルファイルを保存するパス．
    monitor: 監視する値．
    verbose: 冗長モード, 0 または 1．
    save_best_only: save_best_only=Trueの場合，監視しているデータによって最新の最良モデルが上書きされません．
    mode: {auto, min, max}の内の一つが選択されます．save_best_only=Trueならば，現在保存されているファイルを上書きするかは，監視されている値の最大化か最小化によって決定されます．val_accの場合，この引数はmaxとなり，val_lossの場合はminになります．autoモードでは，最大化・最小化のいずれかを監視されている値の名前から自動的に推定します．
    save_weights_only: Trueなら，モデルの重みが保存されます (model.save_weights(filepath))，そうでないなら，モデルの全体が保存されます (model.save(filepath))．
    period: チェックポイント間の間隔（エポック数）．
#### EarlyStopping  
    keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
監視する値の変化が停止した時に訓練を終了する  

    monitor: 監視する値．
    min_delta: 監視する値について改善として判定される最小変化値．つまり，min_deltaよりも絶対値の変化が小さければ改善していないとみなします．
    patience: ここで指定したエポック数の間（監視する値に）改善がないと，訓練が停止します．
    verbose: 冗長モード．
    mode: {auto, min, max}の内，一つが選択される
#### RemoteMonitor  
    keras.callbacks.RemoteMonitor(root='http://localhost:9000', path='/publish/epoch/end/', field='data', headers=None)
サーバーにイベントをストリームするときに使用される  
requestsライブラリが必要.  

    root: 文字列；対象サーバのルートURL．
    path: 文字列；イベントを送るrootへの相対パス．
    field: 文字列；データを保存するJSONのフィールド．
    headers: 辞書; オプションでカスタムできるHTTPヘッダー．
#### LearningRateScheduler  
    keras.callbacks.LearningRateScheduler(schedule, verbose=0)
学習率のスケジューラ．  

    schedule: この関数はエポックのインデックス（整数, 0から始まるインデックス）を入力とし，新しい学習率（浮動小数点数）を返します．
    verbose: 整数．0:：何も表示しない．1：更新メッセージを表示．
#### TensorBoard  
    keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
Tensorboardによる基本的な可視化．  
TensorBoardのログを出力  
TensorBoardでは，異なる層への活性化ヒストグラムと同様に，訓練とテストの評価値を動的にグラフ化し，可視化できる  

    log_dir: TensorfBoardによって解析されたログファイルを保存するディレクトリのパス
    histogram_freq: モデルの層の活性化ヒストグラムを計算する（エポック中の）頻度．この値を0に設定するとヒストグラムが計算されません．ヒストグラムの可視化にはバリデーションデータを指定しておく必要があります．
    write_graph: TensorBoardのグラフを可視化するか．write_graphがTrueの場合，ログファイルが非常に大きくなることがあります．
    write_grads: TensorBoardに勾配のヒストグラフを可視化するかどうか．histogram_freqは0より大きくしなければなりません．
    batch_size: ヒストグラム計算のネットワークに渡す入力のバッチサイズ．
    write_images: TensorfBoardで可視化するモデルの重みを画像として書き出すかどうか．
    embeddings_freq: 選択したembeddingsレイヤーを保存する（エポックに対する）頻度．
    embeddings_layer_names: 観察するレイヤー名のリスト．もしNoneか空リストなら全embeddingsレイヤーを観察します．
    embeddings_metadata: レイヤー名からembeddingsレイヤーに関するメタデータの保存ファイル名へマップする辞書． メタデータのファイルフォーマットの詳細． 全embeddingsレイヤーに対して同じメタデータファイルを使う場合は文字列を渡します．
#### ReduceLROnPlateau  
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
評価値の改善が止まった時に学習率を減らす.  
モデルは訓練が停滞した時に学習率を2〜10で割ることで恩恵を受けることがある．  
このコールバックは評価値を監視し， 
'patience'で指定されたエポック数の間改善が見られなかった場合，学習率を減らす． 

    monitor: 監視する値．
    factor: 学習率を減らす割合．new_lr = lr * factor
    patience: 何エポック改善が見られなかったら学習率の削減を行うか．
    verbose: 整数．0: 何も表示しない．1: 学習率削減時メッセージを表示．
    mode: auto，min，maxのいずれか．  minの場合，監視する値の減少が停止した際に，学習率を更新します．  maxの場合，監視する値の増加が停止した時に，学習率を更新します．  autoの場合，監視する値の名前から自動で判断します．  

    epsilon: 改善があったと判断する閾値．有意な変化だけに注目するために用います．
    cooldown: 学習率を減らした後，通常の学習を再開するまで待機するエポック数．
    min_lr: 学習率の下限．
#### CSVLogger  
    keras.callbacks.CSVLogger(filename, separator=',', append=False)
各エポックの結果をcsvファイルに保存する  

    np.ndarrayのような1次元イテラブルを含む，文字列表現可能な値をサポートしている.
    filename: csvファイル名．例えば'run/log.csv'．
    separator: csvファイルで各要素を区切るために用いられる文字．
    append: True: ファイルが存在する場合，追記します．（訓練を続ける場合に便利です） False: 既存のファイルを上書きします．
#### LambdaCallback  
    keras.callbacks.LambdaCallback(on_epoch_begin=None, on_epoch_end=None, on_batch_begin=None, on_batch_end=None, on_train_begin=None, on_train_end=None)
シンプルな自作コールバックを急いで作るためのコールバック．  
適切なタイミングで呼び出される無名関数で構築されます．  
以下のような位置引数が必要  
    on_epoch_beginとon_epoch_endは2つの位置引数が必要です: epoch，logs  
    on_batch_beginとon_batch_endは2つの位置引数が必要です: batch，logs  
    on_train_beginとon_train_endは1つの位置引数が必要です: logs  

    on_epoch_begin: すべてのエポックの開始時に呼ばれます．
    on_epoch_end: すべてのエポックの終了時に呼ばれます．
    on_batch_begin: すべてのバッチの開始時に呼ばれます．
    on_batch_end: すべてのバッチの終了時に呼ばれます．
    on_train_begin: 訓練の開始時に呼ばれます．
    on_train_end: 訓練の終了時に呼ばれます．

[keras-callback](https://keras.io/ja/callbacks/)



# Class  
## ImageDataGenerator  
リアルタイムにデータ拡張しながら，テンソル画像データのバッチを生成します．  
また，このジェネレータは，データを無限にループするので，無限にバッチを生成します．  

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


        
