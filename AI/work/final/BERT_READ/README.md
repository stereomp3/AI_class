**本筆記為參考李宏毅老師的影片撰寫，還有其他參考資料如下**

https://youtu.be/UYPa347-DdE

https://www.youtube.com/watch?v=ugWDIIOHtPA

https://hackmd.io/@shaoeChen/Bky0Cnx7L

https://andy6804tw.github.io/2021/05/03/ntu-multi-head-self-attention/

[論文連結_Deep contextualized word representations](https://arxiv.org/abs/1802.05365)

[論文連結_Layer Normalization](https://arxiv.org/abs/1607.06450)



# Encoding

最早讓電腦讀懂人類文字是使用稀疏矩陣，每一個文字代表不同符號

1-of-N Encoding

```
apple    = [1 0 0 0 0]
bag      = [0 1 0 0 0]
cat      = [0 0 1 0 0]
dog      = [0 0 0 1 0]
elephant = [0 0 0 0 1]
```

這樣詞彙之間沒有任何的關聯性



所以有word class的概念出現

```
dog cat bird 
ran jumped walk
flower tree apple
```

但是像是最上面的動物，就沒辦法區分更細(哺乳類和鳥類...)



所以就有word embedding出現了，每一個詞彙都使用向量去表示，向量維度就代表這個詞彙的某種意義，語意相近的詞彙(使用上下文訓練相關性)，向量會比較接近

![](H:/SeniorProject02/neuralNetwork/transform/picture/wordEmbedding.png)



在訓練的時候是使用 word embedding的feature訓練，可以廣泛應用在NLP(自然語言學習)上面

同一個詞彙會有不同意義，在過去的做法中，會把詞彙向量(word embedding)分成不同部分，讓一個詞彙會有它意義的向量數目，但這樣往往不夠，因為有些字他的意思可能在時代中又多出了幾個



## ELMO

Embeddings from Language Model (ELMO)

但現在所以要分成 tokens(分類詞彙語意)和type(詞彙)，每一個token都有他自己的embedding，上下文越相近的，就會有越相近的embedding，這個技術又稱為contextualized word embedding

ELMO就是使用到這個技術，他是RNN-based language models (由很多句子訓練出來的)，不需要標註很多句子，就可以訓練

ELMO使用RNN可以判斷出下一個詞彙應該要接甚麼

```
潮水 退了 就 知道 誰 沒穿 褲子

潮水   退了    就
 ↥      ↥      ↥
RNN -> RNN -> RNN
 ↥      ↥      ↥
===    ===    ===  (embedding)
 ↥      ↥      ↥
RNN -> RNN -> RNN
 ↥      ↥      ↥
<BOS>  潮水   退了
```



輸入後就可以拿到 contextualized word embedding

所以假設上下文不同，拿到的embedding也會不一樣，RNN會根據前文書出embedding

ELMO 為了要根據前後文做判斷，所以他也會倒過來訓練一遍，然後把兩個 contextualized word embedding 組合起來



因為ELMO是deep learning，所以他會有很多層在做這種事情，ELMO在論文中的做法是全部選擇，然後使用weighted sum

![](H:/SeniorProject02/neuralNetwork/transform/picture/ELMO.png)



在不同任務中，設置的權重會不一樣，權重會由輸入參數(a1、a2)，讓模型自行訓練出來

![](H:/SeniorProject02/neuralNetwork/transform/picture/ELMO_output.png)

每個token會通過上面的LSTM結果來把最後token呈現的樣子訓練出來

SRL(Semantic Role Labeling): 語意結構化

Coref: 把代名詞代表的目標找出來(他)

SQuAD: QA的問題



Coref和SQuAD就特別需要第一層的 contextualized word embedding，因此不同的任務就需要抽不同的Contextualized Word Embedding來用

# Transformer

//變形金剛

transformer 就是Seq2seq model with "self-attention"

Seq2seq model : RNN就是一個這樣的模型，在輸出b1~b4(詞向量; word embedding)就已經把a1~a4都看過了

![](H:/SeniorProject02/neuralNetwork/transform/picture/Sequence.png)

但是像是RNN，就很難處裡平行化



## CNN replace RNN

所以就有人想出使用CNN取代RNN

![](H:/SeniorProject02/neuralNetwork/transform/picture/CNN2RNN.png)

使用多層CNN，讓資訊讀取存成sequence，然後再重複動作，就可以讓a1~a4相關聯，而且可以直接平行化一起算



## self-Attention

上面的CNN取代RNN需要很多層，所以self-Attention想要取代他

self-Attention Layer 可以輸入一個sequence (a1~a4)，然後輸出一個sequence (b1~b4)，輸出b1~b4就已經把a1~a4都看過了，而且他可以同時進行計算(平行化)!

![](H:/SeniorProject02/neuralNetwork/transform/picture/Self_Attention.png)

所有可以使用RNN做的Self-attention都可以做，論文已經被洗過一輪了



![](H:/SeniorProject02/neuralNetwork/transform/picture/Self-attention_softmax.png)

上面這張圖大致操作如下

```sh
# a 是 x 乘上他的 word embedding，然後輸入到self-attention layer
ai = W*xi

# 每一個不同的input，都乘上不同的 metric 得到不同的Vector
q: query(match others)
qi = Wq * ai

k: key(to be matched)
ki = Wk * ai

v: information to be extracted
vi = Wv * ai

# 拿每一個 q 去對每一個 k 做 attention
a(1,1): q1對k1的attention
...
a(1,4): q1對k4的attention

# attention就是吃兩個向量，吐出一個分數，下方是公式，
# *是做向量內積，d 是 q1跟ki dimension(維度)
# 架設q1和ki內的維度越高，那出來的值就會越大，這邊除d就是防止a值過大
a(1,i) = (q1 * ki) / (d ^ 1/2)  

# 最後再做soft-max，對所有的值做標準化
```



![](H:/SeniorProject02/neuralNetwork/transform/picture/Self-attention_b1.png)

最後把所有值做 weighted sum，就會得到b1 vector。可以看到

b1有用到所有值的資訊(a1~a4)

如果今天想考慮句子部分資訊，就可以把部分最後產生出來的a'變成0，像是如果只要考慮a1、a4，就可以把a'(1,2)和a'(1,3)設為0



self-Attenion很好做平行化，因為他可以把全部的輸入轉成矩陣做運算

```sh
[q1, q2, q3, q4] = Wq * [a1, a2, a3, a4]
--> Q = Wq * I

[k1, k2, k3, k4] = Wk * [a1, a2, a3, a4]
--> K = Wk * I

[v1, v2, v3, v4] = Wv * [a1, a2, a3, a4]
--> V = Wv * I
```



```sh
a(1,1) = k1 * q1
a(1,2) = k1 * q1
a(1,3) = k1 * q1
a(1,4) = k1 * q1
--> [a(1,1), a(1,2), a(1,3), a(1,4) = [k1, k2, k3, k4] * q1
--> A = K * Q
```

上面的運算可以變成2維直接把a(1,1)到a(4,4)做處裡，直接把這個矩陣使用 `[k1, k2, k3, k4] * [q1, q2, q3, q4]`算出來



最後再把A使用Softmax轉為A'，然後與 `[v1, v2, v3, v4]`相乘，就可以得到`[b1, b2, b3, b4]`

```
--> A' = S * A
--> O  = V * A'
```



總而言之，就如同下圖

![](H:/SeniorProject02/neuralNetwork/transform/picture/Self-attention.png)

這些就是一堆矩陣運算，可以使用GPU進行加速



> Muti-head self-attention

下面是self-attention的變形Muti-head self-attention

![](H:/SeniorProject02/neuralNetwork/transform/picture/Muti_head_Self-attention.png)

做的事情差不多，不過最後會把取得的元素做降為運算，得到新的值

```
[b(i,1), b(i,2)] WO = bi
```

這個方法的好處是，不同的head可以關注在不同事情上(關注local、關注globel)，得到更好的輸出



position encoding

在旁邊的詞彙跟在很遠的詞彙是一樣的，沒有順序之分，所以必須要再加上一個神奇的vector(unique positional vector) ei，人工設定，把ei與ai做連接，ei是一個one-hot-vector，1的位置就代表目前位置

<img src="H:/SeniorProject02/neuralNetwork/transform/picture/PositionEncoder.png" style="zoom:60%;" />



# BERT

Bidirectional Encoder Representation from Transformers(BERT) = Encoder of Transformer

BERT其實就是一個非監督式學習的Transformer

不需要資料標記，只需要一大堆句子，BERT主要是接收句子，每個句子會給出word embeding

每個詞彙會對應到一個word embedding，所以一個句子會對應到一串word embedding

中文輸入盡量使用"字"，不要使用"詞"，因為詞的數量太多了，會讓word embedding的維度太大



可以使用兩種方法訊練BERT，同時使用兩個例子，會學的最好

1. Mask LM: 克漏字填空，會蓋掉一個句子裡面15%的詞彙，然後BERT會猜測詞彙的字是什麼

   BERT會把詞彙丟入 Linear Multi-class Classifiter(線性分類器)做預測，類別的數量就是詞彙的數量

   如果兩個詞彙填在同一個地方沒有違和感，那他們就有類似的embedding

   ```
   潮水 [MASK] 就 知道
   ```

   

2. Next Sentence Prediction:下一個句子的預測

   給定兩個句子，讓BERT判斷兩句是否可以接在一起

   ```
   醒醒 吧 [SEP] 你 沒有 妹妹
   ```

   訓練的資料，可以在兩個句子之間加上[SEP]，讓BERT了解這是兩個句子

   ```
   [CLS] 醒醒 吧 [SEP] 你 沒有 妹妹
   ```

   還要在開頭加上[CLS]，讓BERT可以從這個點輸出word embedding並丟入到 Linear Binary Classifier 做輸出(yes、No)，判斷兩句是否可以接在一起



BERT可以解決的問題有4種: 

1. 輸入句子，輸出類型

   ![](H:/SeniorProject02/neuralNetwork/transform/picture/ClassifyBERT.png)

   每一個詞彙都會丟出一個embedding

   只有線性模型需要從頭開始訓練

2. 輸入句子，輸出各個詞彙的類別

   ![](H:/SeniorProject02/neuralNetwork/transform/picture/EachClassifyBERT.png)

   每一個詞彙都做類別判斷，一樣使用線性模型

3. 輸入兩個句子，判斷類別

   ![](H:/SeniorProject02/neuralNetwork/transform/picture/SentencesClassifyBERT.png)

   這邊是使用輸入前提(premise)和假設(hypothesis)，讓機器判斷在這個前提下，這個假設正不正確還是不知道，

4. 給定文章，問問題，給答案(文章有的字才可以)

   文章有N個token，問題有M個token，輸入文章和問題，輸出兩個整數 (s, e)代表文章中的位置 s 到 e

   ![](H:/SeniorProject02/neuralNetwork/transform/picture/QABERT.png)

   紅色和藍色的vector都是訓練出來的，用來回答問題

   紅色與藍色的dimension vector與黃色的都一樣，紅色與黃色做dot product在經過softmax，數值最高的就是s，藍色也一樣。

   如果s和e矛盾了(s=3, e=2)，就會回答次題無解

   ![](H:/SeniorProject02/neuralNetwork/transform/picture/SentencesClassifyBERT02.png)



BERT 一共有24層，每一層都在做NLP相關的工作，如下圖

![](H:/SeniorProject02/neuralNetwork/transform/picture/BERTLayer.png)

bert在input方面做得是比較文法類的任務，靠近output任務是比較複雜的任務

可以從抽出來的數據看到，那些任務會比較需要哪些層，像是POS就會比較需要BERT的第11~13層的word embedding weight sum

後面比較複雜的Relation、Coref... 就比較需要BERT後面層數

相對於ELMO的94M參數，BERT有340M的參數