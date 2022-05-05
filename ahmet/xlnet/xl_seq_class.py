
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
from twitter_data import TwitterDataset

torch.cuda.empty_cache()

# number of epochs to train
# on my local machine I didn't even finish 1 epoch so higher epoch counts would mainly be beneficial for use in clusters
EPOCHS = 3

n_data = []
p_data = []

# read positive and negative tweet data
# use own path here
with open('C:/Users/Ahmet/ETH_Master/FS 22/ICL/twitter_ds/twitter-datasets/train_neg.txt', encoding='utf-8') as f:
    lines = f.read().splitlines()
    n_data = lines
print('read negative tweet lines ', len(n_data))

with open('C:/Users/Ahmet/ETH_Master/FS 22/ICL/twitter_ds/twitter-datasets/train_pos.txt', encoding='utf-8') as f:
    lines = f.read().splitlines()
    p_data = lines
print('read positive tweet lines ', len(p_data))

# fill all tweet data and target labels
tweet_data = p_data + n_data
tweet_labels = [1.0 for _ in p_data] + [0.0 for _ in n_data]
print(len(tweet_data))
print(len(tweet_labels))


# -------------------------------------------------
'''
preprocessing
'''
# -------------------------------------------------

tweet_data = preprocess.remove_punct(tweet_data)
tweet_data = preprocess.remove_stopwords(tweet_data)
tweet_data = preprocess.remove_non_word_char(tweet_data)
tweet_data = preprocess.remove_digits(tweet_data)
tweet_data = preprocess.remove_user_url(tweet_data)

print('preprocess done! First 10 elements: ', tweet_data[:10])

# -------------------------------------------------
'''
train model
'''
# -------------------------------------------------

# use gpu for faster training
print('cuda available: ', torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# use base xlnet model. a larger model may give better results
PRE_TRAINED_MODEL_NAME = 'xlnet-base-cased'
tokenizer = XLNetTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=2)
model = model.to(device)

# char length of longest tweet
max_tweet_len = np.amax([len(t) for t in tweet_data])

# split into training and testing data
tw_train, tw_test, y_train, y_test = train_test_split(tweet_data, tweet_labels, test_size=0.2)

# create instance of dataloader class for tweets
def create_data_loader(data, targets, tokenizer, max_len, batch_size, n_workers=0):
    ds = TwitterDataset(
        tweets=data,
        targets=np.array(targets),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(ds, batch_size=batch_size, num_workers=n_workers)


# data loader for training data
train_data_loader = create_data_loader(tw_train, y_train, tokenizer, max_len=max_tweet_len, batch_size=8, n_workers=0)
# data loader for test data
test_data_loader = create_data_loader(tw_test, y_test, tokenizer, max_len=max_tweet_len, batch_size=8, n_workers=0)

# parameter optimization
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
                                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay':0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)

total_steps = len(train_data_loader) * EPOCHS

# use scheduler for optimizer
scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)


# train a single epoch
def train_epoch(model, data_loader, optimizer, device, scheduler, step=100):
    '''
    train model for one epoch
    :param model: model to be trained. in our case this is the base xlnet model
    :param data_loader: data loader
    :param optimizer: optimizer
    :param device: cpu ("cpu") or gpu ("cuda:0") device
    :param scheduler: scheduler
    :param step: how many training steps until accuracy gets reported for the last "step" steps
    :return: mean accuracy and loss
    '''
    model = model.train()
    losses = []
    acc = 0
    counter = 0

    for d in data_loader:
        # continue
        input_ids = d["input_ids"].reshape(8, max_tweet_len).to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask, labels=targets)
        loss = outputs[0]
        logits = outputs[1]

        # preds = preds.cpu().detach().numpy()
        _, prediction = torch.max(outputs[1], dim=1)
        targets = targets.cpu().detach().numpy()
        prediction = prediction.cpu().detach().numpy()
        accuracy = accuracy_score(targets, prediction)

        acc += accuracy
        losses.append(loss.item())

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        counter += 1

        if counter % step == 0:
            print('iteration: ', counter, '; acc: ', acc / step)
            print(targets)
            print(prediction)
            acc = 0

    return acc / counter, np.mean(losses)


# similar to train_epoch
def eval_model(model, data_loader, device):  # , n_examples):
    model = model.eval()
    losses = []
    acc = 0
    counter = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].reshape(8, max_tweet_len).to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask, labels=targets)
            loss = outputs[0]
            logits = outputs[1]

            _, prediction = torch.max(outputs[1], dim=1)
            targets = targets.cpu().detach().numpy()
            prediction = prediction.cpu().detach().numpy()
            accuracy = accuracy_score(targets, prediction)

            acc += accuracy
            losses.append(loss.item())
            counter += 1

    return acc / counter, np.mean(losses)


history = defaultdict(list)
best_accuracy = 0

# train for epochs
for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    # training function
    train_acc, train_loss = train_epoch(
        model,
        train_data_loader,
        optimizer,
        device,
        scheduler
    )

    print(f'Train loss {train_loss} Train accuracy {train_acc}')

    val_acc, val_loss = eval_model(
        model,
        test_data_loader,
        device
    )

    print(f'Val loss {val_loss} Val accuracy {val_acc}')
    print()

    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)

