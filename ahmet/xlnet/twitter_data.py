
import numpy as np
import matplotlib.pyplot as plt
import nltk
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from transformers import XLNetTokenizer, XLNetModel, XLNetForSequenceClassification, AdamW,\
    get_linear_schedule_with_warmup
import preprocess
from keras.preprocessing.sequence import pad_sequences

'''
class for twitter dataset is subclass of pytorch Dataset
for use with data loader
'''

class TwitterDataset(Dataset):

    def __init__(self, tweets, targets, tokenizer, max_len):
        self.tweets = tweets
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, item):
        tweet = str(self.tweets[item])
        target = self.targets[item]

        # create encoding
        encoding = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=False,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = pad_sequences(encoding['input_ids'], maxlen=self.max_len, dtype=torch.Tensor, truncating="post",
                                  padding="post")
        input_ids = input_ids.astype(dtype='int64')
        input_ids = torch.tensor(input_ids)

        attention_mask = pad_sequences(encoding['attention_mask'], maxlen=self.max_len, dtype=torch.Tensor,
                                       truncating="post", padding="post")
        attention_mask = attention_mask.astype(dtype='int64')
        attention_mask = torch.tensor(attention_mask)

        return {
            'tweet': tweet,
            'input_ids': input_ids,
            'attention_mask': attention_mask.flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


def create_data_loader(data, targets, tokenizer, max_len, batch_size, n_workers=0):
    '''
    create dataloader from dataset
    :param data: (tweet) data
    :param targets: label data (0 and 1 for us)
    :param tokenizer: tokenizer used for model
    :param max_len: max sequence length (max tweet len)
    :param batch_size: ...
    :param n_workers: num workers
    :return:
    '''
    ds = TwitterDataset(
        tweets=data,
        targets=np.array(targets),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(ds, batch_size=batch_size, num_workers=n_workers)

