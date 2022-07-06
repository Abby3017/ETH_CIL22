
import numpy as np
import pandas as pd
from preprocess import preprocess_tweets, tokens_to_sentence, delete_duplicates
from sentence_transformers import SentenceTransformer, util
import time
from numpy import dot
from numpy.linalg import norm
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import os

rs = 42

n_data = []
p_data = []

# read positive and negative tweet data
# use own path here
with open('C:/Users/Ahmet/ETH_Master/FS 22/CIL/twitter_ds/twitter-datasets/train_neg.txt', encoding='utf-8') as f:
    lines = f.read().splitlines()
    n_data = lines
print('read negative tweet lines ', len(n_data))

with open('C:/Users/Ahmet/ETH_Master/FS 22/CIL/twitter_ds/twitter-datasets/train_pos.txt', encoding='utf-8') as f:
    lines = f.read().splitlines()
    p_data = lines

print('read positive tweet lines ', len(p_data))

# fill all tweet data and target labels
tweet_data = p_data + n_data
tweet_labels = [1.0 for _ in p_data] + [0.0 for _ in n_data]
print(len(tweet_data))
print(len(tweet_labels))

strt = time.time()
X_train, X_test, Y_train, Y_test = train_test_split(tweet_data, tweet_labels, test_size=0.1, random_state=42)
print('preprocessing took: ', time.time()-strt, 's ')

'''
strt = time.time()
u_x_train, u_y_train = delete_duplicates(X_train, Y_train)
print(len(u_x_train))
print('preprocessing took: ', time.time()-strt, 's ')
'''

strt = time.time()
print(len(X_train), X_train[:10])
print(len(Y_train), Y_train[:10])
# u_x_train = list(dict.fromkeys(X_train, Y_train).keys())
u_xy_dict = dict(zip(X_train, Y_train))
u_x_train = list(u_xy_dict.keys())
u_y_train = list(u_xy_dict.values())
print(len(u_x_train), u_x_train[:10])
print(len(u_y_train), u_y_train[:10])
print('len after duplicate deletion: ', len(u_x_train))
print('deleting duplicates took: ', time.time()-strt)

'''
with open('C:/Users/Ahmet/ETH_Master/FS 22/CIL/twitter_ds/twitter-datasets/unique_train.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(u_x_train))

with open('C:/Users/Ahmet/ETH_Master/FS 22/CIL/twitter_ds/twitter-datasets/small_x_train.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(X_train))

with open('C:/Users/Ahmet/ETH_Master/FS 22/CIL/twitter_ds/twitter-datasets/unique_y.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join([str(y) for y in u_y_train]))

with open('C:/Users/Ahmet/ETH_Master/FS 22/CIL/twitter_ds/twitter-datasets/small_y_train.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join([str(y) for y in Y_train]))
'''

# print(X_train[:10])
# do duplicate deletion
# sorted(set(X_train), key=X_train.index)

exit()

# do preprocessing

print('\n\nlength before preprocessing: ', len(X_train))
for xt in X_train[-100:-90]:
    print(xt)

# preprocess tweets
X_train, X_test, Y_train = preprocess_tweets(X_train, X_test, Y_train)

print('\n\n length after preprocessing: ', len(X_train))
for xt in X_train[-100:-90]:
    print(xt)

exit()

print('\n\n')
print(X_train[-3:])
print(X_test[-3:])
X_train = tokens_to_sentence(X_train)
print(X_train[-3:])
X_test = tokens_to_sentence(X_test)
print(X_test[-3:])


# cutoff for testing embeddings and knn
cutoff = 10000

# how many positive and negative tweets in first 10k
Y_train = Y_train[:cutoff]
print('total positive out of 10k tweets: ', sum(Y_train))

# sentence transformer
# make sentence embeddings for shuffle_doc_words
model = SentenceTransformer('all-mpnet-base-v2')
strt_train = time.time()
X_train_embeddings = model.encode(X_train[:cutoff], batch_size=32, show_progress_bar=True, convert_to_numpy=True)
print('tweet embeddings done in ', time.time()-strt_train, 's, length of embeddings: ', len(X_train_embeddings))
strt_test = time.time()
X_test_embeddings = model.encode(X_test, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
print('tweet embeddings done in ', time.time()-strt_test, 's, length of embeddings: ', len(X_test_embeddings))

XY_train = []
for i, _ in enumerate(X_train_embeddings):
    XY_train.append((Y_train[i], X_train_embeddings[i]))

pred = []

# randomized kNN
for i, xt in enumerate(X_test_embeddings):
    if i % 1000 == 0:
        print('iteration: ', i)
    # randomly select 1000 tweets to compare with
    XY_samples = random.sample(XY_train, 1000)
    cos_vals = []
    for xy in XY_samples:
        # compute cosine similarity
        cos_sim = dot(xt, xy[1]) / (norm(xt) * norm(xy[1]))
        cos_vals.append((xy[0], cos_sim))
    cos_vals.sort(key=lambda tup: tup[1])
    assert cos_vals[-1][1] > cos_vals[0][1], 'first value of sorted list has to be smaller than last value'
    # do weighted knn classification
    pred.append(1.0 if sum([cv[1] * ((i + 1) * 0.1) for i, cv in enumerate(cos_vals[-10:])]) >
                       sum([i*0.1 for i in range(1, 11)])/2.0 else 0.0)
    if i < 20:
        print([cv[1] * ((i + 1) * 0.1) for i, cv in enumerate(cos_vals[-10:])], pred[-1], Y_test[i])

print('first 10 test: ', Y_test[:10])
print('first 10 pred: ', pred[:10])
print(classification_report(Y_test, pred))
