# model C:\Users\user\.cache\huggingface\hub

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertConfig
import random
from CostemModel import ClassifierModel
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit  # 使用Cross-validation
import numpy as np
from sklearn import metrics


class DataProcessor(object):
    # data process
    def __init__(self, data_file, tokenizer, max_seq_len, label2id):
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.label2id = label2id
        self.data = self._read_data()

    def _read_data(self):
        data = []
        d_index = 0
        for d in self.data_file:
            d_index += 1
            count = 0
            for k in self.data_file[d]:
                if d_index == 1:
                    data.append({"label": self.label2id[k]})
                if d_index == 2:
                    data[count]["sent"] = k
                count += 1
        # random.shuffle(data)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):  # 根據你的資料作調整..  # set it for you own data
        sent = self.data[index]["sent"]
        label = self.data[index]["label"]
        # !!
        features = self.tokenizer(sent, padding=True, truncation=True,
                                  max_length=self.max_seq_len)  # ,add_special_token=True)

        padding_len = self.max_seq_len - len(features["input_ids"])
        input_ids = features["input_ids"] + [self.tokenizer.pad_token_id] * padding_len
        token_type_ids = features["token_type_ids"] + [0] * padding_len
        attention_mask = features["attention_mask"] + [0] * padding_len
        return {"input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
                "label": label}


def collate_fn(batch_data):  # 手動將取出來的資料堆疊
    input_ids = [item["input_ids"] for item in batch_data]
    token_type_ids = [item["token_type_ids"] for item in batch_data]
    attention_mask = [item["attention_mask"] for item in batch_data]
    label = [item["label"] for item in batch_data]

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    label = torch.tensor(label, dtype=torch.long)
    return {"input_ids": input_ids, "token_type_ids": token_type_ids,
            "attention_mask": attention_mask, "label": label}


# 重新訓練 !!!
def main():
    model_name = "bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained("./output/")  # build model
    data = pd.read_csv("./data/disease_data.csv", on_bad_lines='skip')  # 讀自己的Data
    # data = pd.read_csv("./data/train_data.csv", on_bad_lines='skip')
    label2id = {}
    labels = []
    for i in data.id:
        labels.append(i)
    li_lables = list(set(labels))
    li_lables.sort()
    for v, k in enumerate(li_lables):
        label2id[k] = v
    id2label = {value: key for key, value in label2id.items()}

    # data_file = "./data/train_data.csv"
    batch_size = 10  # 一次丟幾筆資料
    epochs = 2  # 完整資料要餵幾次
    lr = 1e-5
    max_seq_len = 128
    device = "cuda"  # cuda (use gpu)  # cpu
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_dataset = DataProcessor(data, tokenizer, max_seq_len, label2id)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=False, collate_fn=collate_fn)

    config = BertConfig.from_pretrained(model_name, num_labels=len(label2id),
                                        id2label=id2label, label2id=label2id)

    model = ClassifierModel.from_pretrained(model_name, config=config,
                                            num_class=len(label2id))
    model.to(device)
    optimizer = AdamW(params=model.parameters(), lr=lr)
    # train
    times = 0
    loss_f = 0
    while True:
        model.train()
        # enumerate 會列出資料的編號 (step, data) 像是
        for step, batch_data in enumerate(train_dataloader):
            input_ids = batch_data["input_ids"].to(device)
            token_type_ids = batch_data["token_type_ids"].to(device)
            attention_mask = batch_data["attention_mask"].to(device)
            labels = batch_data["label"].to(device)

            loss, _ = model(input_ids, token_type_ids, attention_mask, labels)
            optimizer.zero_grad()
            loss.backward()  # 反傳遞
            loss_f = loss
            optimizer.step()
            print("step:{}, loss:{}".format(step + 1, loss))
        if loss_f < 0.1:
            break
        model.save_pretrained("./output/")  # build mode
        times += 1
        print("epoch time: ", times)


main()
