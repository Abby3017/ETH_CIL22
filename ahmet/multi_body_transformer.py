
from transformers import AutoTokenizer, AutoModel, ElectraTokenizer, ElectraModel
from transformers.modeling_outputs import TokenClassifierOutput
import numpy as np
import torch.nn as nn
import torch

class TwoBodyModel(nn.Module):

    def __init__(self, model_1, model_2, model_1_size, model_2_size, num_labels):
        super(TwoBodyModel, self).__init__()
        # electra model has output size 256, distilbert has 768
        self.num_labels = num_labels
        self.model_1 = AutoModel.from_pretrained(model_1)
        self.model_2 = AutoModel.from_pretrained(model_2)
        self.dropout_1 = nn.Dropout(0.2)
        self.dropout_2 = nn.Dropout(0.2)
        self.pre_classifier_1 = nn.Linear(model_1_size, model_1_size)
        self.pre_classifier_2 = nn.Linear(model_2_size, model_2_size)
        self.dropout_1_2 = nn.Dropout(0.1)
        self.dropout_2_2 = nn.Dropout(0.1)
        self.classifier_1 = nn.Linear(model_1_size+model_2_size, model_1_size+model_2_size)
        self.dropout_3 = nn.Dropout(0.2)
        self.classifier_2 = nn.Linear(model_1_size+model_2_size, num_labels)

    def forward(self, input_ids_1, input_ids_2, attention_mask_1, attention_mask_2, labels):
        #  model(input_ids=input_ids, attention_mask=attention_mask, labels=targets)
        output_1 = self.model_1(input_ids=input_ids_1, attention_mask=attention_mask_1)
        output_2 = self.model_2(input_ids=input_ids_2, attention_mask=attention_mask_2)
        # print(output_1[0].shape)
        # print(output_2[0].shape)
        # use: [CLS] token, we can obtain it by typing outputs[0][:, 0, :]
        pre_output_1 = self.dropout_1(output_1[0][:, 0, :])
        pre_output_2 = self.dropout_2(output_2[0][:, 0, :])
        # print(pre_output_1.shape)
        # print(pre_output_2.shape)
        pre_output_1 = self.pre_classifier_1(pre_output_1)
        pre_output_1 = self.dropout_1_2(pre_output_1)
        pre_output_2 = self.pre_classifier_2(pre_output_2)
        pre_output_2 = self.dropout_2_2(pre_output_2)
        output = self.classifier_1(torch.cat((pre_output_1, pre_output_2), 1))
        # print(output.shape)
        output = self.dropout_3(output)
        logits = self.classifier_2(output)
        # print(logits.shape)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # exit()

        return loss, logits




