
import numpy as np
import nltk
from nltk.corpus import stopwords, names
import re

nltk.download('stopwords')

def remove_punct(data):
    return [re.sub('\s+', ' ', d).lower() for d in data]

def remove_names(data):
    # load female and male names
    m_names = names.words('male.txt')
    f_names = names.words('female.txt')
    m_names_lower = [m.lower() for m in m_names]
    f_names_lower = [f.lower() for f in f_names]
    all_names = m_names_lower + f_names_lower
    # iterate over data and only remove words which are names
    for i, _ in enumerate(data):
        data[i] = ' '.join([w for w in data[i].split() if not (w in all_names)])
    return data

def remove_stopwords(data):
    # english stopwords
    sw = stopwords.words('english')
    # iterate over data and only remove words which are stopwords
    for i, _ in enumerate(data):
        data[i] = ' '.join([w for w in data[i].split() if not (w in sw)])
    return data

def remove_non_word_char(data):
    return [re.sub('\W+', ' ', d) for d in data]

def remove_user_url(data):
    # iterate over data and only remove words which are 'user' or 'url'
    for i, _ in enumerate(data):
        data[i] = ' '.join([w for w in data[i].split() if not (w in ['user', 'url'])])
    return data

def remove_digits(data):
    return [re.sub('\d+', '', d) for d in data]

