import logging
import time

import numpy as np
import torch
import torch.nn as nn
import torchtext.functional as F
import torchtext.transforms as T
from torch.hub import load_state_dict_from_url
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterableWrapper
from torchtext.models import XLMR_BASE_ENCODER, RobertaClassificationHead

logging.basicConfig(level=logging.INFO, filename='roberta'+ str(time.time()) +'.log',
                    filemode='w', format='%(name)s - %(levelname)s - %(message)s')


DEVICE = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

padding_idx = 1
bos_idx = 0
eos_idx = 2
max_seq_len = 256
xlmr_vocab_path = r"https://download.pytorch.org/models/text/xlmr.vocab.pt"
xlmr_spm_model_path = r"https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model"

text_transform = T.Sequential(
    T.SentencePieceTokenizer(xlmr_spm_model_path),
    T.VocabTransform(load_state_dict_from_url(xlmr_vocab_path)),
    T.Truncate(max_seq_len - 2),
    T.AddToken(token=bos_idx, begin=True),
    T.AddToken(token=eos_idx, begin=False),
)

file_path_pos = "/cluster/home/abkumar/dataset/twitter-datasets/train_pos.txt"
file_path_neg = "/cluster/home/abkumar/dataset/twitter-datasets/train_neg.txt"


def train_step(input, target):
    output = model(input)
    target = target.long()
    loss = criteria(output, target)
    optim.zero_grad()
    loss.backward()
    optim.step()


def eval_step(input, target):
    output = model(input)
    target = target.long()
    loss = criteria(output, target).item()
    return float(loss), (output.argmax(1) == target).type(torch.float).sum().item()


def evaluate(dev_loader):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    counter = 0
    with torch.no_grad():
        for batch in dev_loader:
            input = F.to_tensor(batch['token_ids'],
                                padding_value=padding_idx).to(DEVICE)
            target = torch.tensor(batch['target']).to(DEVICE)
            loss, predictions = eval_step(input, target)
            total_loss += loss
            correct_predictions += predictions
            total_predictions += len(target)
            counter += 1

    return total_loss / counter, correct_predictions / total_predictions


def get_train_loader(train_datapipe, batch_size=16):
    train_datapipe = train_datapipe.map(lambda x: (text_transform(x[0]), x[1]))
    train_datapipe = train_datapipe.batch(batch_size)
    train_datapipe = train_datapipe.rows2columnar(["token_ids", "target"])
    train_dataloader = DataLoader(train_datapipe, batch_size=None)
    return train_dataloader


def get_dev_loader(dev_datapipe, batch_size=16):
    dev_datapipe = dev_datapipe.map(lambda x: (text_transform(x[0]), x[1]))
    dev_datapipe = dev_datapipe.batch(batch_size)
    dev_datapipe = dev_datapipe.rows2columnar(["token_ids", "target"])
    dev_dataloader = DataLoader(dev_datapipe, batch_size=None)
    return dev_dataloader


def get_train_test(file_path_pos, file_path_neg):
    data_pos = []
    data_neg = []
    with open(file_path_pos) as f:
        for i in f:
            t = i.replace('<user>', '')
            t1 = t.replace('<url>', '')
            data_pos.append((t1, int(1)))

    with open(file_path_neg) as f:
        for i in f:
            t = i.replace('<user>', '')
            t1 = t.replace('<url>', '')
            data_neg.append((t1, int(0)))

    data = data_pos + data_neg
    np_data = np.array(data, dtype='U500, i4')
    N = len(np_data)
    np.random.seed(20)
    shuffler = np.random.permutation(N)
    Ntrain = 150000
    X_train = np_data[shuffler[:Ntrain]]
    X_test = np_data[shuffler[Ntrain:]]
    return X_train, X_test


if __name__ == "__main__":

    X_train, X_test = get_train_test(file_path_pos, file_path_neg)
    train_datapipe = IterableWrapper(X_train)
    dev_datapipe = IterableWrapper(X_test)

    train_loader = get_train_loader(train_datapipe)
    dev_loader = get_dev_loader(dev_datapipe)

    num_classes = 2
    input_dim = 768

    classifier_head = RobertaClassificationHead(
        num_classes=num_classes, input_dim=input_dim)
    model = XLMR_BASE_ENCODER.get_model(head=classifier_head)
    model.to(DEVICE)

    learning_rate = 1e-5
    optim = AdamW(model.parameters(), lr=learning_rate)
    criteria = nn.CrossEntropyLoss()

    num_epochs = 10

    for e in range(num_epochs):
        for batch in train_loader:
            input = F.to_tensor(batch['token_ids'],
                                padding_value=padding_idx).to(DEVICE)
            target = torch.tensor(batch['target']).to(DEVICE)
            train_step(input, target)

        loss, accuracy = evaluate(dev_loader)
        print("Epoch = [{}], loss = [{}], accuracy = [{}]".format(
            e, loss, accuracy))
        logging.info("Epoch = [{}], loss = [{}], accuracy = [{}]".format(
            e, loss, accuracy))
