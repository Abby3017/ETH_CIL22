
from preprocess import preprocess_tweets, tokens_to_sentence
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from wordsegment import load as load_segment

rs = 42

n_data = []
p_data = []

# read positive and negative tweet data
# use own path here!
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

# do train/test split
X_train, X_test, Y_train, Y_test = train_test_split(tweet_data, tweet_labels, test_size=0.1, random_state=42)

# do tweet preprocessing
# training and test data are list of strings and labels are a list of floats
strt = time.time()
# IMPORTANT: this has to be done if we want hastag segmentation, otherwise ignore the next line
load_segment()
X_train, X_test, Y_train = preprocess_tweets(X_train, X_test, Y_train, digit_flag=True, common_flag=True,
                                             spellcheck_flag=True, duplicate_flag=False, segmentation=True,
                                             interlabel_thresh=0.05, user_url_flag=False, rare=10)
print('preprocessing took ', str(time.time()-strt), ' s')

print('\n\n length of training set after preprocessing: ', len(X_train))

print('last 100 training elements: \n', X_train[-100:])
print('last 100 testing elements: \n', X_test[-100:])


'''
thre rest is simply a sample application of tfidf + svc on the preprocessed data
'''

# transform train and test tweets back to list of sentences
# TfidfVectorizer takes list of strings ['this is tweet 1', 'here is tweet 2']
# NOT list of lists of tokens [['this', 'is', 'tweet', '1'], ['here', 'is', 'tweet', '2']]
X_train = tokens_to_sentence(X_train)
X_test = tokens_to_sentence(X_test)


# tfidf vectorizer
vectorizer = TfidfVectorizer()
# transform both train and test set
tfidf_vecs = vectorizer.fit_transform(X_train+X_test)
print(tfidf_vecs.get_shape())
X_train, X_test = tfidf_vecs[:len(X_train), :], tfidf_vecs[len(X_train):, :]
print('len tfidf x train: ', X_train.shape)

# do linear svc
svc = LinearSVC()
svc.fit(X_train, Y_train)
print(svc.score(X_test, Y_test))
