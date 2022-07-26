#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
from transformers import AutoTokenizer, AutoModel
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import Dataset, TensorDataset, DataLoader
# from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import pad_sequences
from multi_body_transformer import TwoBodyModel
import gc
import time
import warnings
import os
import time
from preprocess import preprocess_tweets, tokens_to_sentence
import csv

os.environ['TRANSFORMERS_OFFLINE'] = "1"
os.environ['HF_DATASETS_OFFLINE'] = "1"

bs = 8
EPOCHS = 2
tr = 1  # 8  # 512
# split into separate datasets

# torch.cuda.init()

# torch.cuda.empty_cache()

# print(torch.cuda.memory_summary(0))
# print(torch.cuda.memory_reserved(0))

print('cuda available: ', torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('batch size: ', bs)

# exit()

# model_name_1 = './distilbert-base-uncased-finetuned-sst-2-english'  # 'distilroberta-base' 'google/electra-small-discriminator'
# model_name_1 = 'distilbert-base-uncased'
model_name_1 = "./distilbert-sst/model"
tokenizer_name_1 = "./distilbert-sst/tokenizer"

model_size_1 = 768
model_name_2 = 'google/electra-small-discriminator'
model_size_2 = 256

# xlnet_tokenizer = XLNetTokenizer.from_pretrained(xlnet_model)
# xlnet_tokenizer = XLNetTokenizer.from_petrained(xlnet_model_name)
tokenizer_1 = AutoTokenizer.from_pretrained(tokenizer_name_1)
tokenizer_2 = AutoTokenizer.from_pretrained(model_name_2)

# read positive and negative tweet data
# use own path here!
with open('../../../data/train_neg_full.txt', encoding='utf-8') as f:
    lines = f.read().splitlines()
    n_data = lines
print('read negative tweet lines ', len(n_data))

# '../../../data/train_neg.txt'

with open('../../../data/train_pos_full.txt', encoding='utf-8') as f:
    lines = f.read().splitlines()
    p_data = lines

print('read positive tweet lines ', len(p_data))

# load test data
with open('../../../data/test_data.txt', encoding='utf-8') as f:
    lines = f.read().splitlines()
    t_data = lines

print('read test tweets lines ', len(t_data))

t_data = [t[t.find(',')+1:] for t in t_data]
# print(t_data[:10], t_data[-10:], len(t_data))
# exit()

# fill all tweet data and target labels
tweet_data = p_data + n_data
# tweet_labels = [1.0 for _ in p_data] + [0.0 for _ in n_data]
tweet_labels = [1 for _ in p_data] + [0 for _ in n_data]
print(len(tweet_data))
print(len(tweet_labels))

# tweet_data, _, tweet_label
# 8192  # 4096  # 2048

# do train/test split
t_size = 0.1
if tr == 1:
    t_size = 0.01
X_train, X_test, Y_train, Y_test = train_test_split(tweet_data, tweet_labels, test_size=t_size, random_state=42)

X_train, X_test, Y_train, Y_test = X_train[:int(len(X_train)/tr)], X_test[:int(len(X_test)/tr)], Y_train[:int(len(Y_train)/tr)], Y_test[:int(len(Y_test)/tr)]

print('data length (train, test, train, test): ', len(X_train), len(X_test), len(Y_train), len(Y_test))

print(X_train[:5])
print(Y_train[:5])

# exit()

# do tweet preprocessing
# training and test data are list of strings and labels are a list of floats
strt = time.time()
# IMPORTANT: this has to be done if we want hastag segmentation, otherwise ignore the next line
# load_segment()

X_train_pp, X_test_pp, Y_train_pp = preprocess_tweets(X_train, X_test, Y_train, digit_flag=False, common_flag=False, spellcheck_flag=False, rare_flag=False, duplicate_flag=True, segmentation=False, interlabel_thresh=0.05, user_url_flag=False, rare=5)

print('first preprocessing took: ', time.time()-strt, ' s')

strt = time.time()

# preprocess test tweets
_, real_test, _ = preprocess_tweets(X_train, t_data, Y_train, digit_flag=False, common_flag=False, spellcheck_flag=False, duplicate_flag=False, rare_flag=False, interlabel_thresh=0.05, user_url_flag=False, rare=5)

X_train, X_test, Y_train = X_train_pp, X_test_pp, Y_train_pp

print('second preprocessing took: ', time.time()-strt, ' s')

X_train = tokens_to_sentence(X_train)
X_test = tokens_to_sentence(X_test)
real_test = tokens_to_sentence(real_test)

print('data length (train, test, train, test, real_test): ', len(X_train), len(X_test), len(Y_train), len(Y_test), len(real_test))
warnings.warn('data length (train, test, train, test, real_test): '+str(len(X_train))+', '+str(len(X_test))+', '+str(len(Y_train))+', '+str(len(Y_test))+', '+str(len(real_test)))

print(real_test[:10])

# exit()

# def __init__(self, model_1, model_2, model_1_size, model_2_size, num_labels):
model = TwoBodyModel(model_name_1, model_name_2, model_size_1, model_size_2, 2)
# model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
# model = AutoModel.from_pretrained(model_name)
model = model.to(device)

max_tweet_len = np.amax([len(t) for t in X_train+X_test])


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

        encoding = self.tokenizer.encode_plus(
        tweet,
        add_special_tokens=True,
        max_length=self.max_len,
        return_token_type_ids=False,
        pad_to_max_length=False,
        return_attention_mask=True,
        return_tensors='pt',
        )

        input_ids = pad_sequences(encoding['input_ids'], maxlen=self.max_len, dtype=torch.Tensor ,truncating="post",padding="post")
        input_ids = input_ids.astype(dtype = 'int64')
        input_ids = torch.tensor(input_ids) 

        attention_mask = pad_sequences(encoding['attention_mask'], maxlen=self.max_len, dtype=torch.Tensor ,truncating="post",padding="post")
        attention_mask = attention_mask.astype(dtype = 'int64')
        attention_mask = torch.tensor(attention_mask)       

        return {
        'tweet': tweet,
        'input_ids': input_ids,
        'attention_mask': attention_mask.flatten(),
        'targets': torch.tensor(target, dtype=torch.long)
        }


def create_data_loader(data, targets, tokenizer, max_len, batch_size, n_workers=0):
    ds = TwitterDataset(
        tweets=data,
        targets=np.array(targets),
        tokenizer=tokenizer,
        max_len=max_len
    )
    
    return DataLoader(ds, batch_size=batch_size, num_workers=n_workers)

train_data_loader_1 = create_data_loader(X_train, Y_train, tokenizer_1, max_len=max_tweet_len, batch_size=bs)
train_data_loader_2 = create_data_loader(X_train, Y_train, tokenizer_2, max_len=max_tweet_len, batch_size=bs)
test_data_loader_1 = create_data_loader(X_test, Y_test, tokenizer_1, max_len=max_tweet_len, batch_size=bs)
test_data_loader_2 = create_data_loader(X_test, Y_test, tokenizer_2, max_len=max_tweet_len, batch_size=bs)

# dummy target for eval
dummy_target = [1 for rt in real_test]

real_data_loader_1 = create_data_loader(real_test, dummy_target, tokenizer_1, max_len=max_tweet_len, batch_size=bs)
real_data_loader_2 = create_data_loader(real_test, dummy_target, tokenizer_2, max_len=max_tweet_len, batch_size=bs)

# print(dir(model))

param_optimizer = list(model.named_parameters())

'''
for name, param in param_optimizer:
    print(name)
    if "11" in name or "10" in name or name in other_params:
        opt_params.append((name, param))
        print('----> taken!', name)
        time.sleep(0.4)

param_optimizer = opt_params
'''

# for n, p in param_optimizer:
#     print(n, p.shape, type(p))

# print(dir(model))
# print(model)

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
                                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay':0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)

total_steps = len(train_data_loader_1) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)

def train_epoch(model, data_loader_1, data_loader_2, optimizer, device, scheduler):  # , n_examples):
    model = model.train()
    losses = []
    total_acc = 0
    total_counter = 0
    acc = 0
    counter = 0
    strt = time.time()
    
    for i, d in enumerate(zip(data_loader_1, data_loader_2)):
        d_1, d_2 = d[0], d[1]
        # input_ids_1 = d_1["input_ids"].reshape(bs, max_tweet_len).to(device)
        input_ids_1 = d_1["input_ids"].reshape(d_1["input_ids"].shape[0], max_tweet_len).to(device)
        attention_mask_1 = d_1["attention_mask"].to(device)
        targets_1 = d_1["targets"].to(device)

        # input_ids_2 = d_2["input_ids"].reshape(bs, max_tweet_len).to(device)
        input_ids_2 = d_2["input_ids"].reshape(d_2["input_ids"].shape[0], max_tweet_len).to(device)
        attention_mask_2 = d_2["attention_mask"].to(device)
        targets_2 = d_2["targets"].to(device)

        # assert targets_1 == targets_2, 'target labels of both data loaders should be the same'

        # strt_output = time.time()
        # outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask, labels=targets)
        # orward(self, input_ids_1, input_ids_2, attention_mask_1, attention_mask_2, targets):
        outputs = model(input_ids_1=input_ids_1, input_ids_2=input_ids_2, attention_mask_1=attention_mask_1,
                        attention_mask_2=attention_mask_2, labels=targets_1)
        # output_times += time.time()-strt_output
        loss = outputs[0]
        logits = outputs[1]

        _, prediction = torch.max(outputs[1], dim=1)
        targets = targets_1.cpu().detach().numpy()
        prediction = prediction.cpu().detach().numpy()
        accuracy = accuracy_score(targets, prediction)

        acc += accuracy
        total_acc += accuracy
        losses.append(loss.item())

        # strt_backward = time.time()
        loss.backward()
        # backward_time += time.time()-strt_backward

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # strt_opt_sched = time.time()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        # opt_sched_time += time.time()-strt_opt_sched
        counter += 1
        total_counter += 1
        
        if i % 1000 == 0 and i > 0:
            print('iteration ', i, '; acc: ', acc / counter)
            warnings.warn('iteration '+str(i)+'; acc: '+str(acc/counter)+'; time: '+str(time.time()-strt)+' s')
            acc = 0
            '''
            print('avg model output time: ', output_times / counter)
            output_times = 0
            print('avg backward time: ', backward_time / counter)
            backward_time = 0
            print('avg optimizer sched time: ', opt_sched_time / counter)
            opt_sched_time = 0
            '''
            counter = 0
            print('time: ', time.time()-strt, ' s')
            strt = time.time()
            # print('input ids 1 shape: ', d_1["input_ids"].shape)
            # print('input ids 2 shape: ', d_2["input_ids"].shape)


    return total_acc / total_counter, np.mean(losses)


def eval_model(model, data_loader_1, data_loader_2, device):  #, n_examples):
    model = model.eval()
    losses = []
    acc = 0
    counter = 0
    preds = []
    logit_array = []
    '''
    for i, d in enumerate(zip(data_loader_1, data_loader_2)):
        d_1, d_2 = d[0], d[1]
        input_ids_1 = d_1["input_ids"].reshape(bs, max_tweet_len).to(device)
        attention_mask_1 = d_1["attention_mask"].to(device)
        targets_1 = d_1["targets"].to(device)

        input_ids_2 = d_2["input_ids"].reshape(bs, max_tweet_len).to(device)
        attention_mask_2 = d_2["attention_mask"].to(device)
        targets_2 = d_2["targets"].to(device)
    '''
    
    with torch.no_grad():
        for i, d in enumerate(zip(data_loader_1, data_loader_2)):
            d_1, d_2 = d[0], d[1]
            # print('input ids 1 shape: ', d_1["input_ids"].shape)
            # print('input ids 2 shape: ', d_2["input_ids"].shape)
            # input_ids_1 = d_1["input_ids"].reshape(bs, max_tweet_len).to(device)
            input_ids_1 = d_1["input_ids"].reshape(d_1["input_ids"].shape[0], max_tweet_len).to(device)
            attention_mask_1 = d_1["attention_mask"].to(device)
            targets_1 = d_1["targets"].to(device)

            # input_ids_2 = d_2["input_ids"].reshape(bs, max_tweet_len).to(device)
            input_ids_2 = d_2["input_ids"].reshape(d_2["input_ids"].shape[0], max_tweet_len).to(device)
            attention_mask_2 = d_2["attention_mask"].to(device)
            targets_2 = d_2["targets"].to(device)
            
            # outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask, labels=targets)
            # outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=targets)
            outputs = model(input_ids_1=input_ids_1, input_ids_2=input_ids_2, attention_mask_1=attention_mask_1, attention_mask_2=attention_mask_2, labels=targets_1)
            loss = outputs[0]
            logits = outputs[1]

            _, prediction = torch.max(outputs[1], dim=1)
            targets = targets_1.cpu().detach().numpy()
            prediction = prediction.cpu().detach().numpy()
            accuracy = accuracy_score(targets, prediction)
            preds += list(prediction.flatten())

            acc += accuracy
            losses.append(loss.item())
            counter += 1

            # print(logits.cpu().detach().numpy().shape)
            # warnings.warn('logits shape: '+str(logits.cpu().detach().numpy().shape))

            logit_array += list(logits.cpu().detach().numpy())

    return acc / counter, np.mean(losses), preds, np.array(logit_array)

preds = []
real_preds = []

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(
        model,
        train_data_loader_1,
        train_data_loader_2,
        optimizer,
        device,
        scheduler
    )

    print(f'Train loss {train_loss} Train accuracy {train_acc}')

    val_acc, val_loss, predictions, val_logits = eval_model(
        model,
        test_data_loader_1,
        test_data_loader_2,
        device
    )

    _, _, real_predictions, real_logits = eval_model(
        model,
        real_data_loader_1,
        real_data_loader_2,
        device
    )

    warnings.warn('val loss: '+str(val_loss)+'; valuation acc: '+str(val_acc))

    print('tweets: ', X_test[:10])
    print('pred: ', predictions[:10])
    print('real: ', Y_test[:10])

    warnings.warn('logits shape: '+str(real_logits.shape))
    np.save('electra_distbert_sst2_gelu_'+str(tr)+'_'+str(epoch)+'_val_logits', val_logits)
    np.save('electra_distbert_sst2_gelu_'+str(tr)+'_'+str(epoch)+'_real_logits', real_logits)

    preds = [p if p==1 else -1 for p in predictions]
    real_preds = [p if p==1 else -1 for p in real_predictions]

    # write test data to csv
    with open('electra_distbert_sst2_gelu_'+str(tr)+'_'+str(epoch)+'_predictions.csv', 'w', newline='') as f:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        for j, p in enumerate(real_preds):
            writer.writerow({'Id':j+1, 'Prediction':p})

    print(f'Val loss {val_loss} Val accuracy {val_acc}')

    torch.save(model, 'electra_distbert_sst2_gelu_model_'+str(tr)+'_'+str(epoch)+'.pt')

'''
with open(model_name_1+'_'+model_name_2+'_'+str(tr)+'_'+str(EPOCHS)+'_predictions.csv', 'w', newline='') as f:
    fieldnames = ['Id', 'Prediction']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for j, p in enumerate(preds):
        writer.writerow({'Id':j, 'Prediction':p})
'''

