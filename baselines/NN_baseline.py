from gensim.models import Word2Vec
from keras.layers import Embedding
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
import tensorflow as tf
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
x_train, x_test, y_train = preprocess_tweets(x_train, x_test, y_train, rare_flag=False, digit_flag=False,
                                             common_flag=False, spellcheck_flag=False, duplicate_flag=False,
                                             segmentation=False, interlabel_thresh=0.05, user_url_flag=False, rare=5,
                                             emoji=False, conjunction=False)

print('preprocessing took ', str(time.time()-strt), ' s')



'''
ATTENTION: Use split into tokens in case of no preprocessing!
'''
#x_train = split_into_tokens(x_train)
#x_test = split_into_tokens(x_test)


#Embedding
print('Word2vec construction')
strt = time.time()
model = Word2Vec(x_train, size=150, window=10, workers=8, min_count=1)
# summarize vocabulary size in model
words = list(model.wv.vocab)
print('W2VEC took: ', time.time()-strt, ' s')
print('Vocabulary size: %d' % len(words))
# save model in ASCII (word2vec) format
#Change the name for each preprocessing step
model.save("W2V_small_42split_rare")

# Set Maximum number of words to be embedded
NUM_WORDS = 25000

print('prepare the training set ...')
strt = time.time()
# Define/Load Tokenize text function
#tokenizer = Tokenizer(num_words=NUM_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
tokenizer = Tokenizer(num_words=NUM_WORDS,lower=True)
# Fit the function on the text
tokenizer.fit_on_texts(x_train)
# Count number of unique tokens
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
# Convert train to sequence
sequences_train = tokenizer.texts_to_sequences(x_train)

# Limit size of train/val to 50 and pad the sequence
X_train = pad_sequences(sequences_train,maxlen=50)
# Convert target to array
Y_train = np.asarray(y_train)
# Printing shape
print('Shape of X train and X validation tensor:', X_train.shape)
print('Shape of label train and validation tensor:', Y_train.shape)
print('preparing training set took: ', time.time()-strt, ' s')


#print('load the model W2VEC')
#model = Word2Vec.load("W2V_small_42split")
word_vectors = model.wv

EMBEDDING_DIM=150
vocabulary_size=min(len(word_index)+1,(NUM_WORDS))

embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))

##MODEL
#Embedding layer
for word, i in word_index.items():
    if i>=NUM_WORDS:
        continue
    try:
        embedding_vector = word_vectors[word]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        vec = np.zeros(EMBEDDING_DIM)
        embedding_matrix[i]=vec
# Define Embedding function using the embedding_matrix
embedding_layer = Embedding(vocabulary_size,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            trainable=True)

model = tf.keras.Sequential([embedding_layer,
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)

])
# Compile the Model
model.compile(loss='binary_crossentropy', optimizer='adam',
          metrics=['acc'])

strt = time.time()
# Train the Model
model.fit(X_train, Y_train, epochs=2, batch_size=25, verbose = 1)
print('Training took: ', time.time()-strt, ' s')

model.save('baselineNN_small_42split_dropout0.5')

#model = tf.keras.models.load_model('baselineNN_small_42split')
print('prepare test set ...')
# Convert train and val to sequence
sequences_test = tokenizer.texts_to_sequences(x_test)
# Limit size of train/val to 50 and pad the sequence
x_test = pad_sequences(sequences_test,maxlen=50)

print('evaluate')
strt = time.time()
# evaluate
loss, acc = model.evaluate(x_test, np.asarray(y_test), verbose=0)
print('Test Accuracy: %f' % (acc*100))
print('Evaluation took: ', time.time()-strt, ' s')


