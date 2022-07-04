
import re


def preprocess_tweets(train_docs, test_docs, train_labels, rare=5, freq_thresh=5, interlabel_thresh=0.05, pos=1.0, neg=0.0):
    """
    preprocess training and test tweets. does the following preprocessing steps:
    1) turns sentences into lists of tokens (words)
    2) removes rare words, occurring less than "rare" times
    3) remove user and url from tweets
    4) remove words consisting solely of digits
    5) remove words that occurr nearly equally frequent for positive and negative tweets
    6) spell-check words that would be discarded otherwise (are rare)
    ------------------------------------------------------------------------
    :param train_docs: training tweets as list of strings
    :param test_docs: testing tweets as list of strings
    :param train_labels: training tweet labels as list of floats
    :param rare: rarity threshold for word frequencies. if below threshold, it is discarded
    :param freq_thresh: frequency threshold for spellchecking. larger values delete more words
    :param interlabel_thresh: interlabel commonality threshold
    :param pos: positive tweet label
    :param neg: negative tweet label
    :return: tuple of preprocessed train tweets, preprocessed test tweets and training labels
    """
    tokenized_train = split_into_tokens(train_docs)
    tokenized_test = split_into_tokens(test_docs)
    # word freq of train and test docs
    word_freq = make_word_freq(tokenized_train + tokenized_test)
    pos_tweets = []
    neg_tweets = []
    # length of tweets and their labels has to be equal
    assert len(train_docs) == len(train_labels)
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
            if not is_user_url(t) and not t.isdigit():
                # do we pass the rare threshold?
                if word_freq[t] > rare:
                    if not common_interlabel(t, pos_word_freq, neg_word_freq, thresh=interlabel_thresh):
                        cleaned_tokens.append(t)
                else:
                    # corrected word candidates
                    word_candidates = spellcheck(t, word_freq, freq_thresh=freq_thresh)
                    for w in word_candidates:
                        # does the corrected word check the other tests?
                        if not is_user_url(w) and not w.isdigit() and not common_interlabel(w, pos_word_freq,
                                                                                            neg_word_freq,
                                                                                            thresh=interlabel_thresh):
                            cleaned_tokens.append(w)
        tokenized_train[i] = cleaned_tokens
    # delete empty elements in training data
    clean_train, clean_label = delete_empties(tokenized_train, train_labels)
    assert len(clean_train) == len(clean_label), "dimension mismatch between train data and labels"
    # iterate over testing data
    for i, tt in enumerate(tokenized_test):
        cleaned_tokens = []
        for j, t in enumerate(tt):
            # is the word not a user, url or digit
            if not is_user_url(t) and not t.isdigit():
                # do we pass the rare threshold?
                if word_freq[t] > rare:
                    if not common_interlabel(t, pos_word_freq, neg_word_freq, thresh=interlabel_thresh):
                        cleaned_tokens.append(t)
                else:
                    # corrected word candidates
                    word_candidates = spellcheck(t, word_freq, freq_thresh=freq_thresh)
                    for w in word_candidates:
                        # does the corrected word check the other tests?
                        if not is_user_url(w) and not w.isdigit() and not common_interlabel(w, pos_word_freq,
                                                                                            neg_word_freq,
                                                                                            thresh=interlabel_thresh):
                            cleaned_tokens.append(w)
        # test tweets shouldn't be empty. if they are, then use the original without preprocessing
        if len(cleaned_tokens) < 1:
            cleaned_tokens = tt
        tokenized_test[i] = cleaned_tokens
    return clean_train, tokenized_test, clean_label


# handle empty training tweets
def delete_empties(train_tweets, tweet_labels):
    # tokenized lists of training tweets that are not empty
    non_empty_train = [t for t in train_tweets if len(t) >= 1]
    # if training tweet not empty label also doesn't get removed
    non_empty_label = [l for i, l in enumerate(tweet_labels) if len(train_tweets[i]) >= 1]
    return non_empty_train, non_empty_label



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

# possibly combine punctuations like ! ? ! into !?!

# word frequency mappin
def make_word_freq(token_lines):
    """
    output word frequency dictionary from token list
    :param token token_lines: list of tokens
    :return: dictionary of word frequencies
    """
    word_freq = {}
    for tl in token_lines:
        for t in tl:
            if t in word_freq:
                word_freq[t] += 1
            else:
                word_freq[t] = 1
    return word_freq

# common words in both positive and negatvie tweets
def common_interlabel(word, p_freq, n_freq, thresh=0.05):
    """
    boolean function, for whether a word is frequent in two separate frequency dicts (usually positive and negative)
    :param word: input word
    :param p_freq: positive frequency dict
    :param n_freq: negative frequency dict
    :param thresh: threshold how far from an equal distribution we can get
    :return: True, if the word is common in both frequency dictionaries
    """
    if word in p_freq and word in n_freq:
        if 1+thresh > (float(p_freq[word])/float(n_freq[word])) > 1-thresh:
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
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
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
