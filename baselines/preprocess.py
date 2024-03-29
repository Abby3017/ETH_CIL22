import re

import nltk
# from hashformers import TransformerWordSegmenter as WordSegmenter
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from wordsegment import load as load_segment
from wordsegment import segment


def preprocess_tweets(train_docs, test_docs, train_labels, rare=5, freq_thresh=5, interlabel_thresh=0.05, pos=1.0,
                      neg=0.0, rare_flag=True, user_url_flag=True, digit_flag=True, common_flag=True,
                      spellcheck_flag=True, duplicate_flag=False, segmentation=False, emoji=False, conjunction=False):
    """
    preprocess training and test tweets. does the following preprocessing steps:
    1) turns sentences into lists of tokens (words)
    2) removes rare words, occurring less than "rare" times
    3) remove user and url from tweets
    4) remove words consisting solely of digits
    5) remove words that occur nearly equally frequent for positive and negative tweets
    6) spell-check words that would be discarded otherwise (are rare)
    7) delete duplicate tweets. disabled by default
    8) do word segmentation to hashtags. disabled by default
    9) replace emojis by token
    10) Apply conjunction rule
    ------------------------------------------------------------------------
    :param train_docs: training tweets as list of strings
    :param test_docs: testing tweets as list of strings
    :param train_labels: training tweet labels as list of floats
    :param rare: rarity threshold for word frequencies. if below threshold, it is discarded
    :param freq_thresh: frequency threshold for spellchecking. larger values delete more words
    :param interlabel_thresh: interlabel commonality threshold
    :param pos: positive tweet label
    :param neg: negative tweet label
    :param rare_flag: if True, does rare word removal
    :param user_url_flag: if True, does common word removal
    :param digit_flag: if True, does digit removal
    :param common_flag: if True, does common word removal
    :param spellcheck_flag: if True, does spellchecking for rare words
    :param duplicate_flag: if True, duplicate tweets get deleted
    :param segmentation: if True, does segmentation of hashtags
    :return: tuple of preprocessed train tweets, preprocessed test tweets and training labels
    """
    # length of tweets and their labels has to be equal
    assert len(train_docs) == len(
        train_labels), 'training tweets and labels length should be the same'
    if duplicate_flag:
        train_docs, train_labels = delete_duplicates(train_docs, train_labels)
    tokenized_train = split_into_tokens(train_docs)
    tokenized_test = split_into_tokens(test_docs)
    if segmentation:
        # do hashtag segmentation
        load_segment()
        tokenized_train = segment_hashtags(tokenized_train)
        tokenized_test = segment_hashtags(tokenized_test)
    if emoji :
        train_docs = [handle_emojis(train_docs[i]) for i in range(len(train_docs))]
        test_docs = [handle_emojis(test_docs[i]) for i in range(len(test_docs))]
    if conjunction:
        tokenized_train = remove_conjunction(tokenized_train)
        tokenized_test = remove_conjunction(tokenized_test)

    # word freq of train and test docs
    word_freq = make_word_freq(tokenized_train + tokenized_test)
    pos_tweets = []
    neg_tweets = []
    # build positive and negative tweet lists
    for i, t in enumerate(tokenized_train):
        if train_labels[i] == pos:
            pos_tweets.append(t)
        elif train_labels[i] == neg:
            neg_tweets.append(t)
        else:
            assert False, "mismatch on tweet label"
    pos_word_freq = make_word_freq(pos_tweets)
    neg_word_freq = make_word_freq(neg_tweets)
    for i, tt in enumerate(tokenized_train):
        cleaned_tokens = []
        for j, t in enumerate(tt):
            # is the word not a user, url or digit
            if (not is_user_url(t) or not user_url_flag) and (not t.isdigit() or not digit_flag):
                # do we pass the rare threshold?
                if word_freq[t] > rare or not rare_flag:
                    if not common_interlabel(t, pos_word_freq, neg_word_freq, thresh=interlabel_thresh) or \
                            not common_flag:
                        cleaned_tokens.append(t)
                elif spellcheck_flag:
                    # corrected word candidates
                    word_candidates = spellcheck(
                        t, word_freq, freq_thresh=freq_thresh)
                    for w in word_candidates:
                        # does the corrected word check the other tests?
                        if (not is_user_url(w) or not user_url_flag) and (not w.isdigit() or not digit_flag) and not \
                                (common_interlabel(w, pos_word_freq, neg_word_freq, thresh=interlabel_thresh) or not
                                common_flag):
                            cleaned_tokens.append(w)
        tokenized_train[i] = cleaned_tokens
    # delete empty elements in training data
    clean_train, clean_label = delete_empties(tokenized_train, train_labels)
    assert len(clean_train) == len(
        clean_label), "dimension mismatch between train data and labels"
    # iterate over testing data
    for i, tt in enumerate(tokenized_test):
        cleaned_tokens = []
        for j, t in enumerate(tt):
            # is the word not a user, url or digit
            if (not is_user_url(t) or not user_url_flag) and (not t.isdigit() or not digit_flag):
                # do we pass the rare threshold?
                if word_freq[t] > rare or not rare_flag:
                    if not common_interlabel(t, pos_word_freq, neg_word_freq, thresh=interlabel_thresh) or \
                            not common_flag:
                        cleaned_tokens.append(t)
                elif spellcheck_flag:
                    # corrected word candidates
                    word_candidates = spellcheck(
                        t, word_freq, freq_thresh=freq_thresh)
                    for w in word_candidates:
                        # does the corrected word check the other tests?
                        if (not is_user_url(w) or not user_url_flag) and (not w.isdigit() or not digit_flag) and not \
                                (common_interlabel(w, pos_word_freq, neg_word_freq, thresh=interlabel_thresh) or not
                                common_flag):
                            cleaned_tokens.append(w)
        # test tweets shouldn't be empty. if they are, then use the original without preprocessing
        if len(cleaned_tokens) < 1:
            cleaned_tokens = tt
        tokenized_test[i] = cleaned_tokens
    # implement duplicate flags at the beginning
    '''
    if duplicate_flag:
        clean_train, clean_label = delete_duplicates(clean_train, clean_label)
    '''
    return clean_train, tokenized_test, clean_label


def segment_hashtags(tokenized_tweets):
    '''
    segments hashtags of tweets
    :param tokenized_tweets: tokenized tweets to do hashtag segmentation on
    :return: hashtag segmented tweets. list of lists of tokens (words)
    '''
    segmented_tweets = []
    for tt in tokenized_tweets:
        tokens = []
        for t in tt:
            tokens.append(t)
            if t[0] == '#' and len(t) > 1:
                # print(type(t[1:]))
                ht_segments = segment(t[1:])
                tokens += ht_segments
        segmented_tweets.append(tokens)
    return segmented_tweets


'''
def hashformers_hashtag_segment(hashtag_words):
    
    segment hashtag in input
    param: pass token starting with hashtag in form of list (but pass word without hashtag)
    return: list of string with segmented hashtags
    
    ws = WordSegmenter(
        # can try other segmenter too more from this list https://huggingface.co/models
        segmenter_model_name_or_path="gpt2",
        reranker_model_name_or_path="bert-base-uncased"  # this is optional
    )
    segmentations = ws.segment(hashtag_words)
    return segmentations
'''

def load_data(pos_path, neg_path):
    '''
    load positive and negative tweets
    :param pos_path: positive tweet path
    :param neg_path: negative tweet path
    :return: list of tweets and list labels
    '''
    # read positive tweets
    with open(pos_path, encoding='utf-8') as f:
        lines = f.read().splitlines()
        n_data = lines
    print('read negative tweet lines ', len(n_data))
    # use negative tweets
    with open(neg_path, encoding='utf-8') as f:
        lines = f.read().splitlines()
        p_data = lines
    print('read positive tweet lines ', len(p_data))
    # fill all tweet data and target labels
    tweet_data = p_data + n_data
    tweet_labels = [1.0 for _ in p_data] + [0.0 for _ in n_data]
    return tweet_data, tweet_labels


def delete_duplicates(tweets, labels):
    '''
    delete duplicates in tweets and output new tweets and labels
    :param tweets: list of tweets
    :param labels: list of labels
    :return: unique list of tweets and matching labels
    '''
    if not len(tweets) == len(labels):
        print('tweet and label length mismatch')
        return tweets, labels
    # make dict of unique tweets and their respective labels
    u_dict = dict(zip(tweets, labels))
    u_tweets = list(u_dict.keys())
    u_labels = list(u_dict.values())
    return u_tweets, u_labels


# handle empty training tweets
def delete_empties(train_tweets, tweet_labels):
    # tokenized lists of training tweets that are not empty
    non_empty_train = [t for t in train_tweets if len(t) >= 1]
    # if training tweet not empty label also doesn't get removed
    non_empty_label = [l for i, l in enumerate(
        tweet_labels) if len(train_tweets[i]) >= 1]
    return non_empty_train, non_empty_label


def lemmatize_sentence(tokens):
    '''
    lemmatization of string
    param tokens: list of token
    return: list of tokens(words)
    '''
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')
    nltk.download('omw-1.4')
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence


def split_into_tokens(doc_lines):
    """
    input a list of strings (representing tweets) and turn them into a list of lists of tokens
    :param doc_lines: list of tweet strings
    :return: list of lists of tokens
    """
    tokenized_lines = []
    for d in doc_lines:
        tokenized_lines.append(d.split())
    return tokenized_lines


# word frequency mapping
def make_word_freq(tokenized_lines):
    """
    output word frequency dictionary from token list
    :param token token_lines: list of tokens
    :return: dictionary of word frequencies
    """
    word_freq = {}
    for tl in tokenized_lines:
        for t in tl:
            if t in word_freq:
                word_freq[t] += 1
            else:
                word_freq[t] = 1
    return word_freq


def common_interlabel(word, p_freq, n_freq, thresh=0.05):
    """
    boolean function, for whether a word is frequent in two separate frequency dicts (usually positive and negative)
    :param word: input word
    :param p_freq: positive frequency dict
    :param n_freq: negative frequency dict
    :param thresh: threshold how far from an equal distribution we can get
    :param lower_bound: lower bound for how frequent the word is
    :return: True, if the word is common in both frequency dictionaries
    """
    if word in p_freq and word in n_freq:
        if 1+thresh > (float(p_freq[word])/float(n_freq[word])) > 1-thresh:
            # print(word)
            return True
    return False

# is user url


def is_user_url(word):
    """
    is the word user or url
    :param word: word in question
    :return: True if the word is user or url
    """
    return word in ['<user>', '<url>', 'user', 'url']

# some spellcheck functions
# edit functions from peter norvig


def edits1(word):
    """
    potentially corrected words created through a single edit
    :param word: word to correct
    :return: list of correction candidates
    """
    "All edits that are one edit away from `word`."
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return deletes + transposes + replaces + inserts


def edits2(word):
    """
    words that are two edits away from word
    :param word: input word to correct
    :return: spellchecked candidates
    """
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


def spellcheck(word, wordfreq, freq_thresh, len_thresh=5, split_words=True):
    """
    spellchecking function
    :param word: word to correct
    :param wordfreq: word frequency dictionary
    :param freq_thresh: how frequent the corrected version of the word must be
    :param len_thresh: if we want to only consider longer corrected words through norvigs spellchecker
    :return: returns a list of spellchecking candidates
    """
    candidates = []
    # delete "-" and seperate words to see if seperately, they are candidates
    if split_words:
        split_candidates = []
        new_words = word.split("-")
        for w in new_words:
            if w in wordfreq and len(w) > 1:
                if wordfreq[w] >= freq_thresh:
                    split_candidates.append(w)
        if len(split_candidates) >= 2:
            return split_candidates
    # check if edit distance sensible and which new words are common enough
    new_words_1 = list(edits1(word))
    # new_words_2 = list(edits2(word))
    max_freq = freq_thresh-1
    for w in new_words_1:  # + new_words_2:
        if w in wordfreq:
            if len(w) >= len_thresh:
                if wordfreq[w] > max_freq:
                    candidates.append(w)
                    max_freq = wordfreq[w]
    # delete all duplicate chars and return potential new word
    single_chars = []
    for i, c in enumerate(word):
        if i < len(word)-1:
            if c != word[i+1]:
                single_chars.append(c)
        else:
            single_chars.append(c)
    new_word = ''.join(single_chars)
    if new_word in wordfreq:
        if wordfreq[new_word] >= freq_thresh:
            candidates.append(new_word)
    # take most frequent word
    max_freq = 0
    max_cand = None
    for c in candidates:
        if wordfreq[c] > max_freq:
            max_cand = c
            max_freq = wordfreq[c]
    if max_cand is None:
        return []
    else:
        return [max_cand]


def tokens_to_sentence(tokens_list):
    """
    outputs list of sentences from tokens
    :param tokens_list: list of lists of tokens
    :return: list of sentences
    """
    sentence_list = []
    for tl in tokens_list:
        sentence_list.append(' '.join(tl))
    return sentence_list

def remove_conjunction(tokenized):
    for i, t in enumerate(tokenized):
        if('but' in t):
            idx = max(index for index, item in enumerate(t) if item == 'but')
            tokenized[i] = t[idx+1:]
        if('although' in t):
            idx = max(index for index, item in enumerate(t) if item == 'although')
            tokenized[i] = t[idx+1:]
        if('however' in t):
            idx = max(index for index, item in enumerate(t) if item == 'however')
            tokenized[i] = t[idx+1:]
    return tokenized

def handle_emojis(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    #tweet = re.sub(r'(:\s?\)|:-\)|\(-:|:\'\))', ' EMO_POS ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    #tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', ' EMO_POS ', tweet)
    tweet = re.sub(r'(:p|=\)|=p|=d|:p|:d|;p|;d|;\)|xx|xxx)', 'EMO_POS', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    #tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    #tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', tweet)
    tweet = re.sub(r'(=\(|:/|\;/)', ' EMO_NEG ', tweet)
    # Cry -- :,(, :'(, :"(
    #tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', tweet)
    return tweet
