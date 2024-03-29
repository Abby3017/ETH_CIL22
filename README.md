# ETH_CIL22 Sentiment Analysis

Implementation of sentiment classification for a Twitter dataset in the scope of the Computational Intelligence Lab @ ETH Zurich, Spring 2022.

# Preprocessing

To preprocess the data we use the `preprocess_tweets()` function in `preprocess.py`. The flags can be set arbitrarily to perform the different preprocessing steps as described in the report.

By default, we only implement duplicate tweet deletion. For hashtag segmentation, the `load_segment()` function from the `wordsegment` library has to be called before `preprocess_tweets()`.

# Usage

We have implemented the classifiers listed below. Run the corresponding files and set the `big_data` variable in each of the files to `True` to train the classifier on the entire dataset; otherwise it will be trained on the small dataset. Make sure you have the files `train_pos.txt`, `train_neg.txt`, `train_pos_full.txt`, `train_pos_neg.txt`, and `test_data.txt` in the `data` folder.

For the distilbert and combined models, the contents of the folder https://polybox.ethz.ch/index.php/s/Q14qxlYeb3oy7ze are needed. Namely the parent folder including the subfolders model and tokenizer subfolders. Make sure this folder is included in the same directory as the classifier python files, as "./distilbert-sst/model" and "./distilbert-sst/tokenizer" need to be accessed.

Each of the classifiers already has preprocessing implemented. The preprocessing methods can be selected by setting the flags in `preprocess_tweets()` in the classifier's file as desired.

### Logistic Regression
Logistic Regression is used as one of the baseline models for this task.
To run this model, run the `logistic_regression_classifier.py` file.

### Bidirectional LSTM
Run `lstm_classifier.py` to train this baseline model.

### ELECTRA Classifier
ELECTRA classifier model for text classification. Parameter big_data should be True (by default) for classification on the full twitter dataset and False for the smaller dataset. Simply run `electra_transformer_classifier.py`, with `preprocess.py` in the same folder and the twitter datasets in the data folder.

### DistilBERT Classifier
DistilBERT Classifier model for text classification. Parameter big_data should be True (by default) for classification on the full twitter dataset and False for the smaller dataset. Simply run `distbert_transformer_classifier.py`, with `preprocess.py` in the same folder and the twitter datasets in the data folder.

### Combined GELU Classifier
Two body transformer classifier model using both ELECTRA and DistilBERT models and a neural network with GELU activation functions as a classification head. Parameter big_data should be True (by default) for classification on the full twitter dataset and False for the smaller dataset. Simply run `multi_classifier.py`, with `preprocess.py` in the same folder and the twitter datasets in the data folder.

Combined GELU, with the `big_data` variable set to `True`, is the exact model we used for the final Kaggle submission. The submission file is the ".csv" written after the second training epoch is done. 

# Requirements

The following packages are required:
- Preprocessing:
    - `nltk`
    - `wordsegment`
- Classifiers:
    - `transformers`
    - `numpy`
    - `scikit-learn`
    - `torch`
    - `tensorflow>=2.9.1`
    - `gensim<=3.8.3`
