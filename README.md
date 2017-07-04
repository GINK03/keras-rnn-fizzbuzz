# KerasのRNNでFizzBuzzを行う(\+ Epochスケジューラの提案)
ディープラーニングをやるようになって半年程度経ちました  
ある程度ならば、文章や画像判別モデルならば、過去の自分の資産をうまく活用することと、外部からState of the Artな手法を導入することで、様々なネットワークを組むことが可能になってまいりました  
しかし、基礎の基礎であるはずの、Fizz Buzzをやるのを忘れていたのです  
やるしかありません  

## 先行研究
- [Fizz Buzz in TensorFlow](http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/)
- [Fizz Buzz Keras](https://github.com/ad34/FizzBuzz-keras)

全結合のモデルでの、Fizz Buzzの評価のようです  

## 提案
RNNでも、FizzBuzzは可能なのではないでしょうか  
全結合層のモデルのみで、1000 ~ 5000程度のデータで学習させることが多いですが、20万件のデータセットで学習させることで、より大きな数字にも対応させることを目標とします  

カリキュラム学習という学習法があり、簡単な問題設定から初めて、徐々に難しくしていくことで、早く安定的に学習できるそうです[1]  

この時、人間がカリキュラムを意図して簡単な問題を用意して学習されるのではなく、学習のデータを最初のうちは限定したデータセットにて学習させ、限定した覚えてきたらデータを拡大し、様々なケースを学習させて、汎化性能を獲得していくという学習方法をとります  

具体的には、データセットとepochにスケジューラを組み込むことで実現します  

## モデル
'1:Fizz, 2:Buzz, 3:Fizz Buzz, 4:そのまま(Path)'と４値の判別問題を全結合層２層でといている問題設定が多いが、
'1:Fizz, 2:Buzz, 3:Path'の３値のそれぞれのアクティブな状態を求める問題設定とする  
つまり、softmaxの活性化関数を用いるのではなく、sigmoidを三つ用いbinary-crossentropyにて最小化する

<p align="center">
  <img width="450px" src="https://user-images.githubusercontent.com/4949982/27811887-b6b1e360-60a5-11e7-96bb-fa5a328090e9.png">
</p>
<div align="center"> 図1. 使用したモデル </div>

コードはKerasを利用した  
モデルとスケジューラは、非常に小さく、わかりやすいです  
モデル  
```python
inputs       = Input(shape=(10, 11))
encoded1     = Bi( GRU(256, activation='relu') )(inputs)
encoded1     = Dense(512, activation='relu')( encoded1 )
encoded1_1x  = Reshape((1,512,))(encoded1)
decoded      = Dense(3, activation='sigmoid')( Flatten()(encoded1_1x) )
fizzbuzz     = Model(inputs, decoded)
fizzbuzz.compile(optimizer=Adam(), loss='binary_crossentropy')
```
スケジューラ（初期のデータセットは、epochを多く学習し、後半になるにつれ一回のみにスケジューリングしている）
```python
class CURRICULUM:
  EPOCH = [50, 30, 20, 10, 5, 1]
  @staticmethod
  def GET():
    if len(CURRICULUM.EPOCH) > 0:
      return CURRICULUM.EPOCH.pop(0)
    else:
      return 1
...
fizzbuzz.fit(Xs, Ys, epochs=CURRICULUM.GET(), callbacks=[batch_callback])
...
```

## 実験
200,000件のFizz Buzzのデータセットを、[スクリプト](https://github.com/GINK03/keras-rnn-fizzbuzz-on-dev/blob/master/data_utils.py)で作成し、 5000件ずつ、データセットを分割し40個のデータセットを学習させる

この時、スケジューリングモデルAは、任意のデータセットをランダムで選択し、以下のepoch回、学習する  
```console
{ 1回目：50epoch, 2回目：30epoch, 3回目：20epoch, 4回目：10epoch, 5回目：5epoch }
```
このスケジューリングが完了した後は、残りのデータセットを1epochで学習する

スケジューリングモデルBは特にスケジューリングは行わず、全てのデータセットを平等に学習していく。なお、この方法は、全てのデータセットをメモリ上に乗せて順番に学習していく方法と変わらない  

## 評価
スケジューリングモデルA（青）とモデルB（赤）で大きな差がでた 
<p align="center">
  <img width="750px" src="https://user-images.githubusercontent.com/4949982/27813446-f4374526-60b0-11e7-957f-c5ee05c8780a.png">
</p>
<div align="center"> 図2. epochごとのlossの変化 </div>
初期値依存性を考慮しても、この差は大きく、スケジューリングを行うことが、まともに収束するしないなどの差を担っているように思われる

モデルAはテストデータにおける精度100%であった  
モデルBは68%であった  

なお、出力はこのようになっている
左から、入力値、人手による結果、予想値、正解だったかどうか、である（PATHとは、そのまま出力するという意味にしました）
ほぼ100%あっていることが確認できた  
```
    64170 original result = Fizz Buzz , predict result = Fizz Buzz , result = True
      9791 original result = Path , predict result = Path , result = True
     54665 original result = Buzz , predict result = Buzz , result = True
    118722 original result = Fizz , predict result = Fizz , result = True
     97502 original result = Path , predict result = Path , result = True
    186766 original result = Path , predict result = Path , result = True
    153331 original result = Path , predict result = Path , result = True
      7401 original result = Fizz , predict result = Fizz , result = True
    117939 original result = Fizz , predict result = Fizz , result = True
     22732 original result = Path , predict result = Path , result = True
     73516 original result = Path , predict result = Path , result = True
    144774 original result = Fizz , predict result = Fizz , result = True
     32783 original result = Path , predict result = Path , result = True
     67097 original result = Path , predict result = Path , result = True
    116715 original result = Fizz Buzz , predict result = Fizz Buzz , result = True
     21195 original result = Fizz Buzz , predict result = Fizz Buzz , result = True
```

## コード

https://github.com/GINK03/keras-rnn-fizzbuzz-on-dev

テストデータを作成する
```console
$ python3 data_utils.py --step1
```
学習する(全体の8割を学習します)
```console
$ python3 fizzbuzz.py --train
```
予想する(テストデータから予想します)
```console
$ python3 fizzbuzz.py --predict
```

## 感想
データによってはまともに収束してくれないものあり、RNNではその傾向が特に顕著です  
性格にロス率の違いなどを測ったことがなかったのですが、Epochをいじることによって、安定して学習させることができることがあるということでした  

## 参考文献
[1] [Deep Learningの技術と未来](https://www.slideshare.net/beam2d/deep-learning-22544096)
