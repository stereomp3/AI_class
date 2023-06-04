# -*- coding: utf-8 -*-
import torch
from transformers import BertTokenizer, BertModel, BertConfig
from CostemModel import ClassifierModel
import pandas as pd

model_path = "../BERT_MODEL/"
label2id = {'三叉神經痛': 0, '中暑': 1, '乳汁分泌不足': 2, '乳腺炎': 3, '五噎反胃': 4, '休克': 5, '利咽': 6, '前臂痛': 7, '前臂神經痛': 8, '口腔頜面病症': 9, '吐食': 10, '吞嚥不利': 11, '吞酸': 12, '呃逆': 13, '咽喉腫痛': 14, '咽炎': 15, '哮喘': 16, '單純性甲狀腺腫': 17, '失眠': 18, '嬰幼兒腹瀉': 19, '小兒夜啼': 20, '小兒消化不良': 21, '尺神經痛': 22, '心律失常': 23, '心悸': 24, '心痛': 25, '心肌炎': 26, '急性咽喉炎': 27, '急性胰腺炎': 28, '急性腰扭傷': 29, '感冒': 30, '扁桃體炎': 31, '手指不能伸屈': 32, '手指麻木': 33, '手臂紅腫': 34, '手臂紅腫疼痛': 35, '拇指屈肌肌腱炎': 36, '掌指麻痹': 37, '支氣管炎': 38, '昏厥': 39, '昏迷': 40, '橈骨莖突部狹窄性腱鞘炎': 41, '止痙': 42, '毒蛇咬傷': 43, '泄熱': 44, '流行性感冒': 45, '消化不良': 46, '滯產': 47, '無脈症': 48, '煩熱': 49, '牙痛': 50, '產婦宮縮無力': 51, '產後乳少': 52, '疲勞乏力': 53, '瘧疾': 54, '癔病': 55, '癲癇': 56, '發熱': 57, '發燒': 58, '白喉': 59, '白癜風': 60, '百日咳': 61, '目痛': 62, '目赤': 63, '神經衰弱': 64, '精神分裂症': 65, '糖尿病': 66, '結膜炎': 67, '經痛': 68, '耳聾': 69, '耳鳴': 70, '聰耳': 71, '肋間神經痛': 72, '肘腕炎': 73, '肘間神經痛': 74, '肩臂痛': 75, '肺結核': 76, '胃擴張': 77, '胃炎': 78, '胎位不正': 79, '胸痛': 80, '脈管炎': 81, '腕三角軟骨盤損傷': 82, '腕管綜合征': 83, '腕背腱鞘囊腫': 84, '腮腺炎': 85, '膽囊炎': 86, '自汗': 87, '舌強腫痛': 88, '舌肌麻痹': 89, '落枕': 90, '蕁麻疹': 91, '閉經': 92, '開竅': 93, '關節炎': 94, '關節痛': 95, '電光性眼炎': 96, '面神經麻痹': 97, '面肌痙攣': 98, '項強': 99, '頭痛': 100, '頭項強痛': 101, '頭風': 102, '顳頜關節功能紊亂': 103, '食慾減退': 104, '食道痙攣': 105, '鼻炎': 106, '鼻血': 107, '鼻衄': 108, '齒神經痛': 109}
id2label = {value: key for key, value in label2id.items()}

config = BertConfig.from_pretrained(model_path)
bert_model = ClassifierModel.from_pretrained(model_path, config=config, num_class=len(label2id))

# 輸出模型參數的key
# for parms in model.named_parameters():
#     print(parms)
# print(model.state_dict().key())
# print(bert_model.state_dict().key())


# 輸入一個句子，模型做預測
text = "我的頭好痛"
tokenizer = BertTokenizer.from_pretrained(model_path)
features = tokenizer(text, padding=True, truncation=True,
                     max_length=32, add_special_tokens=True)

input_ids = torch.tensor([features["input_ids"]], dtype=torch.long)
token_type_ids = torch.tensor([features["token_type_ids"]], dtype=torch.long)
attention_mask = torch.tensor([features["attention_mask"]], dtype=torch.long)
# bert_output = bert_model(input_ids=input_ids, token_type_ids=token_type_ids,
#                         attention_mask=attention_mask)[1]
# print(bert_output.shape)
output = bert_model(input_ids=input_ids, token_type_ids=token_type_ids,
                        attention_mask=attention_mask)
print(output.shape)

predicted_label = output.argmax(dim=1)
predicted_label_value = predicted_label.item()

print(predicted_label)
print(id2label[predicted_label_value])
#print(output.softmax(dim=1))