import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel


# 自定義forward和加入線性分類層到BERT後面
class ClassifierModel(BertPreTrainedModel):
    # bert classifiter
    def __init__(self, config, num_class):
        super(ClassifierModel, self).__init__(config)
        self.config = config
        self.num_class = num_class
        self.bert = BertModel(config=self.config)
        self.hidden_size = config.hidden_size
        self.classifier = nn.Linear(self.hidden_size, self.num_class)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids,
                           attention_mask=attention_mask)[1]

        logits = self.classifier(output)
        if labels is not None:
            entropy_loss = nn.CrossEntropyLoss()
            loss = entropy_loss(logits, labels)
            return loss, logits
        else:
            return logits