
from transformers import AutoTokenizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
# from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import pad_sequences
from multi_body_transformer import TwoBodyModel
import os
import time
from preprocess import preprocess_tweets, tokens_to_sentence
import csv

# is the full twitter dataset used? Set to False to use the smaller dataset.
big_data = True

if big_data:
    t_size = 0.01
    negative_path = './data/train_neg_full.txt'
    positive_path = './data/train_pos_full.txt'
    # batch size
    bs = 8
else:
    t_size = 0.1
    negative_path = './data/train_neg.txt'
    positive_path = './data/train_pos.txt'
    # batch size
    bs = 16

EPOCHS = 2

os.environ['TRANSFORMERS_OFFLINE'] = "1"
os.environ['HF_DATASETS_OFFLINE'] = "1"

print('cuda available: ', torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('batch size: ', bs)

# distilbert model and tokenizer paths
model_name_1 = "./distilbert-sst/model"
tokenizer_name_1 = "./distilbert-sst/tokenizer"

model_name_2 = 'google/electra-small-discriminator'

# output size of distilbert model
model_size_1 = 768
# output size of electra model
model_size_2 = 256

# tokenizer for model 1 (distilbert)
tokenizer_1 = AutoTokenizer.from_pretrained(tokenizer_name_1)
# tokenizer for model 2 (electra)
tokenizer_2 = AutoTokenizer.from_pretrained(model_name_2)

# read positive and negative tweet data
with open(negative_path, encoding='utf-8') as f:
    lines = f.read().splitlines()
    n_data = lines
print('negative tweet lines: ', len(n_data))

with open(positive_path, encoding='utf-8') as f:
    lines = f.read().splitlines()
    p_data = lines

print('positive tweet lines: ', len(p_data))

# load test data
with open('./data/test_data.txt', encoding='utf-8') as f:
    lines = f.read().splitlines()
    t_data = lines

print('test tweets lines: ', len(t_data))

t_data = [t[t.find(',')+1:] for t in t_data]

# fill all tweet data and target labels
tweet_data = p_data + n_data
tweet_labels = [1 for _ in p_data] + [0 for _ in n_data]

X_train, X_test, Y_train, Y_test = train_test_split(tweet_data, tweet_labels, test_size=t_size, random_state=42)

print('data length (train, test, train, test): ', len(X_train), len(X_test), len(Y_train), len(Y_test))

strt = time.time()

# preprocessing for transformer models are done with duplicate tweet deletion as only preprocessing step
X_train_pp, X_test_pp, Y_train_pp = preprocess_tweets(X_train, X_test, Y_train, digit_flag=False, common_flag=False,
                                                      spellcheck_flag=False, rare_flag=False, duplicate_flag=True,
                                                      segmentation=False, interlabel_thresh=0.05, user_url_flag=False,
                                                      rare=5)

print('first preprocessing took: ', time.time()-strt, ' s')

strt = time.time()

# preprocess test tweets
_, real_test, _ = preprocess_tweets(X_train, t_data, Y_train, digit_flag=False, common_flag=False,
                                    spellcheck_flag=False, duplicate_flag=False, rare_flag=False,
                                    interlabel_thresh=0.05, user_url_flag=False, rare=5)

X_train, X_test, Y_train = X_train_pp, X_test_pp, Y_train_pp

print('second preprocessing took: ', time.time()-strt, ' s')

# turn tweet data back into list of sentences instead of lists of lists of tokens
X_train = tokens_to_sentence(X_train)
X_test = tokens_to_sentence(X_test)
real_test = tokens_to_sentence(real_test)

print('data length (train, test, train, test, real_test): ', len(X_train), len(X_test), len(Y_train), len(Y_test),
      len(real_test))

# instance of model with two transformer bodies connected at a classification head neural network
model = TwoBodyModel(model_name_1, model_name_2, model_size_1, model_size_2, 2)
model = model.to(device)

# max tweet length among training and test (validation) set
max_tweet_len = np.amax([len(t) for t in X_train+X_test])

# twitter dataset class
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

# twitter data loader class
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

# dummy target for testing data to be evaluated for kaggle
dummy_target = [1 for rt in real_test]

real_data_loader_1 = create_data_loader(real_test, dummy_target, tokenizer_1, max_len=max_tweet_len, batch_size=bs)
real_data_loader_2 = create_data_loader(real_test, dummy_target, tokenizer_2, max_len=max_tweet_len, batch_size=bs)

param_optimizer = list(model.named_parameters())

# no weight decay for bias and norm layers
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

optimizer_grouped_parameters = [
                                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay':0.0}
]


optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)

total_steps = len(train_data_loader_1) * EPOCHS

# use linear schedule
scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)

def train_epoch(model, data_loader_1, data_loader_2, optimizer, device, scheduler):
    model = model.train()
    losses = []
    total_acc = 0
    total_counter = 0
    acc = 0
    counter = 0
    strt = time.time()
    
    for i, d in enumerate(zip(data_loader_1, data_loader_2)):
        d_1, d_2 = d[0], d[1]
        input_ids_1 = d_1["input_ids"].reshape(d_1["input_ids"].shape[0], max_tweet_len).to(device)
        attention_mask_1 = d_1["attention_mask"].to(device)
        targets_1 = d_1["targets"].to(device)

        input_ids_2 = d_2["input_ids"].reshape(d_2["input_ids"].shape[0], max_tweet_len).to(device)
        attention_mask_2 = d_2["attention_mask"].to(device)
        targets_2 = d_2["targets"].to(device)
        outputs = model(input_ids_1=input_ids_1, input_ids_2=input_ids_2, attention_mask_1=attention_mask_1,
                        attention_mask_2=attention_mask_2, labels=targets_1)
        loss = outputs[0]

        _, prediction = torch.max(outputs[1], dim=1)
        targets = targets_1.cpu().detach().numpy()
        prediction = prediction.cpu().detach().numpy()
        accuracy = accuracy_score(targets, prediction)

        acc += accuracy
        total_acc += accuracy
        losses.append(loss.item())

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        counter += 1
        total_counter += 1
        
        if i % 1000 == 0 and i > 0:
            print('iteration ', i, '; acc: ', acc / counter)
            acc = 0
            counter = 0
            print('time: ', time.time()-strt, ' s')
            strt = time.time()

    return total_acc / total_counter, np.mean(losses)


def eval_model(model, data_loader_1, data_loader_2, device):
    model = model.eval()
    losses = []
    acc = 0
    counter = 0
    preds = []
    logit_array = []
    
    with torch.no_grad():
        for i, d in enumerate(zip(data_loader_1, data_loader_2)):
            d_1, d_2 = d[0], d[1]
            input_ids_1 = d_1["input_ids"].reshape(d_1["input_ids"].shape[0], max_tweet_len).to(device)
            attention_mask_1 = d_1["attention_mask"].to(device)
            targets_1 = d_1["targets"].to(device)

            input_ids_2 = d_2["input_ids"].reshape(d_2["input_ids"].shape[0], max_tweet_len).to(device)
            attention_mask_2 = d_2["attention_mask"].to(device)
            # don't need targets_2 as they are the same for dataloader 1 and 2

            outputs = model(input_ids_1=input_ids_1, input_ids_2=input_ids_2, attention_mask_1=attention_mask_1,
                            attention_mask_2=attention_mask_2, labels=targets_1)
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

            logit_array += list(logits.cpu().detach().numpy())

    return acc / counter, np.mean(losses), preds, np.array(logit_array)

preds = []
real_preds = []

# train and evaluate for 'EPOCHS' epochs
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

    print(f'Val loss {val_loss} Val accuracy {val_acc}')

    _, _, real_predictions, real_logits = eval_model(
        model,
        real_data_loader_1,
        real_data_loader_2,
        device
    )

    # export logits of validation and real (kaggle) test set
    np.save('electra_distbert_sst2_gelu_'+str(epoch)+'_val_logits', val_logits)
    np.save('electra_distbert_sst2_gelu_'+str(epoch)+'_real_logits', real_logits)

    # change negative tweet labels to -1 instead of 0
    preds = [p if p==1 else -1 for p in predictions]
    real_preds = [p if p==1 else -1 for p in real_predictions]

    # write test data to csv
    with open('electra_distbert_sst2_gelu_'+str(epoch)+'_predictions.csv', 'w', newline='') as f:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        for j, p in enumerate(real_preds):
            writer.writerow({'Id':j+1, 'Prediction':p})
