
from transformers import ElectraForSequenceClassification, ElectraTokenizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import Dataset, TensorDataset, DataLoader
from keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.utils import pad_sequences
import warnings
import os
import time
from preprocess import preprocess_tweets, tokens_to_sentence

# os.environ['TRANSFORMERS_OFFLINE'] = "1"
# os.environ['HF_DATASETS_OFFLINE'] = "1"

bs = 64
EPOCHS = 5
tr = 8  # 512

print('cuda available: ', torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('batch size: ', bs)

model_name = 'google/electra-small-discriminator'
model_size = 256

tokenizer = ElectraTokenizer.from_pretrained(model_name)

# read positive and negative tweet data
# use own path here!
with open('C:/Users/Ahmet/ETH_Master/FS 22/CIL/twitter_ds/twitter-datasets/train_neg_full.txt', encoding='utf-8') as f:
    lines = f.read().splitlines()
    n_data = lines
print('read negative tweet lines ', len(n_data))

with open('C:/Users/Ahmet/ETH_Master/FS 22/CIL/twitter_ds/twitter-datasets/train_pos_full.txt', encoding='utf-8') as f:
    lines = f.read().splitlines()
    p_data = lines

print('read positive tweet lines ', len(p_data))

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

X_train, X_test, Y_train = X_train_pp, X_test_pp, Y_train_pp

print('second preprocessing took: ', time.time()-strt, ' s')

X_train = tokens_to_sentence(X_train)
X_test = tokens_to_sentence(X_test)

print('data length (train, test, train, test, real_test): ', len(X_train), len(X_test), len(Y_train), len(Y_test))
warnings.warn('data length (train, test, train, test, real_test): '+str(len(X_train))+', '+str(len(X_test))+', '+str(len(Y_train))+', '+str(len(Y_test)))

# exit()

model = ElectraForSequenceClassification.from_pretrained(model_name, num_labels=2)
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

train_data_loader = create_data_loader(X_train, Y_train, tokenizer, max_len=max_tweet_len, batch_size=bs)
test_data_loader = create_data_loader(X_test, Y_test, tokenizer, max_len=max_tweet_len, batch_size=bs)

param_optimizer = list(model.named_parameters())

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
                                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay':0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)

total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)

def train_epoch(model, data_loader, optimizer, device, scheduler):
    model = model.train()
    losses = []
    total_acc = 0
    total_counter = 0
    acc = 0
    counter = 0
    strt = time.time()
    
    for i, d in enumerate(data_loader):
        # input_ids_1 = d_1["input_ids"].reshape(bs, max_tweet_len).to(device)
        input_ids = d["input_ids"].reshape(d["input_ids"].shape[0], max_tweet_len).to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        # assert targets_1 == targets_2, 'target labels of both data loaders should be the same'

        # strt_output = time.time()
        # outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask, labels=targets)
        # orward(self, input_ids_1, input_ids_2, attention_mask_1, attention_mask_2, targets):
        outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask, labels=targets)
        # output_times += time.time()-strt_output
        loss = outputs[0]
        logits = outputs[1]

        _, prediction = torch.max(outputs[1], dim=1)
        targets = targets.cpu().detach().numpy()
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
        
        if i % 100 == 0 and i > 0:
            print('iteration ', i, '; acc: ', acc / counter)
            warnings.warn('iteration '+str(i)+'; acc: '+str(acc/counter)+'; time: '+str(time.time()-strt)+' s')
            acc = 0
            counter = 0
            print('time: ', time.time()-strt, ' s')
            strt = time.time()
            # print('input ids 1 shape: ', d_1["input_ids"].shape)
            # print('input ids 2 shape: ', d_2["input_ids"].shape)

    return total_acc / total_counter, np.mean(losses)


def eval_model(model, data_loader, device):
    model = model.eval()
    losses = []
    acc = 0
    counter = 0
    preds = []
    logit_array = []
    with torch.no_grad():
        for i, d in enumerate(data_loader):
            input_ids = d["input_ids"].reshape(d["input_ids"].shape[0], max_tweet_len).to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            
            # outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask, labels=targets)
            # outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=targets)
            outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask, labels=targets)
            loss = outputs[0]
            logits = outputs[1]

            _, prediction = torch.max(outputs[1], dim=1)
            targets = targets.cpu().detach().numpy()
            prediction = prediction.cpu().detach().numpy()
            accuracy = accuracy_score(targets, prediction)
            preds += list(prediction.flatten())

            acc += accuracy
            losses.append(loss.item())
            counter += 1

            logit_array += list(logits.cpu().detach().numpy())

    return acc / counter, np.mean(losses), preds, np.array(logit_array)

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(
        model,
        train_data_loader,
        optimizer,
        device,
        scheduler
    )

    print(f'Train loss {train_loss} Train accuracy {train_acc}')

    val_acc, val_loss, predictions, val_logits = eval_model(
        model,
        test_data_loader,
        device
    )

    warnings.warn('val loss: '+str(val_loss)+'; valuation acc: '+str(val_acc))
    print(f'Val loss {val_loss} Val accuracy {val_acc}')

    torch.save(model, 'electra_model_'+str(tr)+'_'+str(epoch)+'.pt')

