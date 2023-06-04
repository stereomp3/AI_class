Code ref: 

* https://leemeng.tw/attack_on_bert_transfer_learning_in_nlp.html
* !! 介紹源代碼https://zhuanlan.zhihu.com/p/363014957
* bert 解決二分類問題: https://www.gushiciku.cn/pl/pCkS/zh-tw
* DataLoader: https://ithelp.ithome.com.tw/articles/10277163
* 討論區: https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertTokenizerFast
* BERT 多標籤實作: https://medium.com/datamixcontent-lab/nlp-%E8%87%AA%E7%84%B6%E8%AA%9E%E8%A8%80%E8%99%95%E7%90%86-bert-fast-ai-%E5%A4%9A%E6%A8%99%E7%B1%A4%E5%88%86%E9%A1%9E%E6%A8%A1%E5%9E%8B%E5%AF%A6%E4%BD%9C-673eca215263
* 這個就是大神的BERT嗎?: https://github.com/lemuria-wchen/imcs21-cblue
* 論文https://arxiv.org/pdf/2204.08997v2.pdf
* try data: https://github.com/UCSD-AI4H/Medical-Dialogue-System
* 裡面有參考文獻: https://zhuanlan.zhihu.com/p/154527264
* GPT-2 chinese model: https://github.com/Morizeyao/GPT2-Chinese#%E6%A8%A1%E5%9E%8B%E5%88%86%E4%BA%AB
* GPT-2: https://zhuanlan.zhihu.com/p/498677758
* kaggle: https://www.kaggle.com/code/yuval6967/toxic-bert-plain-vanila
* cross_validation: https://ithelp.ithome.com.tw/articles/10197461
* https://blog.csdn.net/weixin_42223207/article/details/125820174



程式碼:

* [BERT.py](BERT.py): 主要訓練函數，我沒有提供訓練資料，所以要跑就跑下面的BERT_test.py
* [CostemModel.py](CostemModel.py): 自定義forward和加上分類器
* [BERT_test.py](BERT_test.py): 使用已經訓練完的模型進行預測