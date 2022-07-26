import nltk
from preprocess import preprocess_tweets, split_into_tokens
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import pandas as pd

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
                                             common_flag=False, spellcheck_flag=False, duplicate_flag=False,
                                             segmentation=False, interlabel_thresh=0.05, user_url_flag=False, rare=5,
                                             emoji=False, conjunction=False)

print('preprocessing took ', str(time.time()-strt), ' s')

def word_score(df_pos, df_neg, word):
    try :
        nb_pos = df_pos[word]
    except :
        nb_pos = 0

    try:
        nb_neg = df_neg[word]
    except :
        nb_neg = 0

    try:
        pos = nb_pos/(nb_pos+nb_neg)
        neg = nb_neg/(nb_pos+nb_neg)
        return pos, neg
    except :
        return 0,0

def tweet_score(tweet, df_pos, df_neg):
    score = 0
    for w in tweet:
        w_score = word_score(df_pos, df_neg, w)[0]
        if (w_score != None and w_score != 0):
            if(w_score<0.4 or w_score>0.6):
                score += w_score*2 -1
    return score

train_pos = []
train_neg = []
for i in range(len(x_train)):
    if(y_train[i]==1):
        train_pos.append(x_train[i])
    else:
        train_neg.append(x_train[i])


print('tagging')
tagged_pos = [nltk.pos_tag(train_pos[i]) for i in range(len(train_pos))]
tagged_neg = [nltk.pos_tag(train_neg[i]) for i in range(len(train_neg))]

pos_list = [x for xs in tagged_pos for x in xs]
neg_list = [x for xs in tagged_neg for x in xs]

pos_word=[]
print('Select positive nouns, verbs, adverbs, ... ')
for word, tag in pos_list:
    if (tag == 'FW' or tag == 'JJ' or tag == 'JJS' or tag == 'JJR' or tag == 'NN' or tag == 'NNS' or tag =='RB' or tag == 'RBR'
        or tag == 'RBS' or tag=='UH' or tag=='VB' or tag == 'VBD' or tag == 'VBG' or tag=='VBN' or tag=='VBP' or tag=='VBZ'):
        if(len(word)<12 and len(word)>1):
            pos_word.append(word)

neg_word=[]
print('Select negative nouns, verbs, adverbs, ... ')
for word, tag in neg_list:
    if (tag == 'FW' or tag == 'JJ' or tag == 'JJS' or tag == 'JJR' or tag == 'NN' or tag == 'NNS' or tag =='RB' or tag == 'RBR'
        or tag == 'RBS' or tag=='UH' or tag=='VB' or tag == 'VBD' or tag == 'VBG' or tag=='VBN' or tag=='VBP' or tag=='VBZ'):
        if(len(word)<12 and len(word)>1):
            neg_word.append(word)

print('Create df with the occurence of each word in the positive and negative sentences')
df_pos = pd.value_counts(pos_word)
df_neg = pd.value_counts(neg_word)

#change name for each preprocessing step 
df_pos.to_csv('voc_pos_small42split.csv')
df_neg.to_csv('voc_neg_small42split.csv')

df_pos = pd.read_csv('voc_pos_small42split.csv')
df_neg = pd.read_csv('voc_neg_small42split.csv')


print('Calculate the score for each test sentence')
y_test_voc = [tweet_score(x_test[i], df_pos, df_neg) for i in range(len(x_test))]
y = [1 if y_test_voc[i]>0 else 0 for i in range(len(y_test_voc))]
print('accuracy: ' ,accuracy_score(y_test, y))
