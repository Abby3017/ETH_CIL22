from preprocess import preprocess_tweets, split_into_tokens
import numpy as np
from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfVectorizer)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import time
from scipy import sparse
from wordsegment import load as load_segment

n_data = []
p_data = []

rs = 42

# read positive and negative tweet data
# use own path here
with open('/Users/louancillon/Documents/ETHZ/M2/CIL/ETH_CIL22/twitter-datasets/train_neg.txt', encoding='utf-8') as f:
    lines = f.read().splitlines()
    n_data = lines
print('read negative tweet lines ', len(n_data))

with open('/Users/louancillon/Documents/ETHZ/M2/CIL/ETH_CIL22/twitter-datasets/train_pos.txt', encoding='utf-8') as f:
    lines = f.read().splitlines()
    p_data = lines
print('read positive tweet lines ', len(p_data))

# fill all tweet data and target labels
tweet_data = p_data + n_data
tweet_labels = [1.0 for _ in p_data] + [0.0 for _ in n_data]
print(len(tweet_data))
print(len(tweet_labels))

x_train, x_test, y_train, y_test = train_test_split(tweet_data, tweet_labels, test_size=0.1, random_state=rs)


# do tweet preprocessing
strt = time.time()
'''
IMPORTANT: load_segment() has to be done if we want hastag segmentation, otherwise comment it out
'''
# load_segment()
x_train, x_test, y_train = preprocess_tweets(x_train, x_test, y_train, rare_flag=False, digit_flag=True,
                                             common_flag=False, spellcheck_flag=True, duplicate_flag=False,
                                             segmentation=True, interlabel_thresh=0.05, user_url_flag=True, rare=5,
                                             emoji=True, conjunction=False)

print('preprocessing took ', str(time.time()-strt), ' s')



'''
ATTENTION: Use split into tokens in case of no preprocessing!
'''
#x_train = split_into_tokens(x_train)
#x_test = split_into_tokens(x_test)



LR = LogisticRegression(
    verbose=0,
    class_weight = "balanced",
    random_state = 0,
    multi_class = "multinomial",
    max_iter=100 ** 2,
)

# Create a pipeline
baseline = Pipeline([
  ("tf-idf", TfidfVectorizer(lowercase=False, tokenizer=lambda i:i)),
   #TfidfVectorizer(lowercase=False)),
  ("clf", LR)
])

strt = time.time()

baseline.fit(X=x_train, y=y_train)
y_pred = baseline.predict(x_test)
acc = accuracy_score(y_test, y_pred)

print('logistic regression took: ', time.time()-strt, ' s')
print('accuracy: ', acc)

