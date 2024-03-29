
from transformers import AutoModel
import torch.nn as nn
import torch
import torch.nn.functional as F

class TwoBodyModel(nn.Module):

    def __init__(self, model_1, model_2, model_1_size, model_2_size, num_labels):
        '''
        init method of the two body transformer model
        :param model_1: name of first model for initialization with AutoModel from the transformers library
        :param model_2: name of the second model for initialization with AutoModel from the transformers library
        :param model_1_size: output size of first model
        :param model_2_size: output size of the second model
        :param num_labels: number of labels for classification
        '''
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
        '''
        forward function for the two body transformer model
        :param input_ids_1: input id's generated by the tokenizer of model 1
        :param input_ids_2: input id's generated by the tokenizer of model 2
        :param attention_mask_1: attention mask generated by the tokenizer of model 1
        :param attention_mask_2: attention mask generated by the tokenizer of model 2
        :param labels: classification labels
        :return: tuple consisting of: Cross entropy loss, Logits from the output of the last layer
        '''
        output_1 = self.model_1(input_ids=input_ids_1, attention_mask=attention_mask_1)
        output_2 = self.model_2(input_ids=input_ids_2, attention_mask=attention_mask_2)
        pre_output_1 = self.dropout_1(F.gelu(output_1[0][:, 0, :]))
        pre_output_2 = self.dropout_2(F.gelu(output_2[0][:, 0, :]))
        pre_output_1 = self.pre_classifier_1(pre_output_1)
        pre_output_1 = self.dropout_1_2(F.gelu(pre_output_1))
        pre_output_2 = self.pre_classifier_2(pre_output_2)
        pre_output_2 = self.dropout_2_2(F.gelu(pre_output_2))
        output = self.classifier_1(torch.cat((pre_output_1, pre_output_2), 1))
        output = self.dropout_3(F.gelu(output))
        logits = self.classifier_2(output)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return loss, logits




