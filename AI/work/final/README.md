**程式碼以及筆記皆為整理網路上的資料出來的，程式碼在針對訓練資料的部分(資料輸入，資料訓練)是原創，訓練BERT的資料集是原創，從維基百科上面整理出來的疾病對應症狀的csv檔案，CSV檔案我這邊就不提供了，我就只給最後訓練完的模型**



資料夾以及文件

* [BERT_MODEL](BERT_MODEL): 最後訓練出來的模型

* [BERT_READ](BERT_READ): 整理BERT相關筆記，參考李宏毅老師的影片

* [BERT_USE](BERT_USE): BERT訓練([BERT.py](BERT_USE/BERT.py)、[CostemModel.py](BERT_USE/BERT.py))以及BERT測試文件([BERT_test.py](BERT_USE/BERT_test.py))



注意: 在跑之前，需要確認電腦設備，至少要顯卡 1650，RAM 24GB以上



BERT是google開發的模型，使用muti head attention，讓詞跟詞之間的關係更容易被學習，我透過fine-turn BERT base模型(參數規模110 M)，在BERT pre-train model後面加上一層分類器，開發了一個從使用者輸入的症狀推測疾病的模型。



下面大概介紹我如何訓練和最後訓練出來的模型指標

# BERT_train

使用pytorch撰寫



​                                                                               bert base                                

![](picture/BERT_base.png)

使用了 BERT 的預訓練模型使用[bert-base-chinese](https://huggingface.co/bert-base-chinese.)，該模型具有12層layer、768層hidden、12個heads，參數規模達到110MB。在進行模型訓練前，我們對 BERT 模型進行token的輸入初始化，並加入我們從維基百科蒐集的資料。目前我們的資料集中包含109種類別的疾病資料，以及與這些疾病可能發生的症狀相關的對話資訊（表一）。總共有623筆資料。我們使用這些資料進行模型的fine-tuning，即在BERT的最後一層接上一個新的簡單的分類器，用於識別使用者輸入的疾病類別。

​                                                                               表一

| **disease** | **text**            |
| ----------- | ------------------- |
| 流行性感冒  | 發燒、咳嗽、喉嚨痛… |
| 急性咽喉炎  | 咽部疼痛，吞嚥疼痛… |
| ....        | ....                |
| 中暑        | 頭暈、頭痛、口渴…   |
| 消化不良    | 持續性食慾不振…     |



​                                                                               BERT 訓練過程

![](picture/BERT_train.png)



我將資料集分為八成的訓練集和兩成的測試集對BERT模型進行fine-turn，在訓練模型時，我們使用epochs為72，batch size為10。當損失函數小於0.1時，我們才結束訓練。



# BERT_metric

​                                                                               BERT 損失函數(使用 cross-entropy)

![](picture/BERT_loss.png)

​                                                                               BERT 指標(accuracy、precision、recall、f1score)

![](picture/BERT_metric.png)

模型表現出色，達到近八成的準確率