import os

import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

PATH = os.path.dirname(os.path.abspath(__file__))


def load_data(vocab_size):
    path = os.path.join(PATH, "raw", "spam-flat3.csv")
    mydata = pd.read_csv(path)
    with_subject = mydata["subject"] + " " + mydata["bodyHtml"]
    with_subject = with_subject.fillna("Ignore")
    X_ws = with_subject.tolist()
    mydata.loc[mydata['phishing'].astype(str) == 'False', 'phishing'] = 0
    mydata.loc[mydata['phishing'].astype(str) == 'True', 'phishing'] = 1
    y = mydata['phishing'].astype(int).tolist()
    X_ws_train, X_ws_test, y_train, y_test = train_test_split(X_ws, y, test_size=0.3, random_state=42)
    trainX = preprocess_text(X_ws_train)
    testX = preprocess_text(X_ws_test)
    trainY = preprocess_label(y_train)
    testY = preprocess_label(y_test)
    return X_ws_train, X_ws_test, trainX, trainY, testX, testY


def stemSentence(sentence):
    token_words = word_tokenize(sentence)
    stem_sentence = []
    for word in token_words:
        stem_sentence.append(PorterStemmer().stem(word))
        stem_sentence.append(" ")
        return "".join(stem_sentence)


def preprocess_text(data_x):
    max_length = 350
    vocab_size = 10000
    X_2 = [stemSentence(x) for x in data_x]
    encoded_texts_ws = [one_hot(t, vocab_size) for t in X_2]
    padded_texts_ws = pad_sequences(encoded_texts_ws, maxlen=max_length, padding="post", truncating="post")
    return padded_texts_ws


def preprocess_label(data_y):
    y_1 = []
    for x in data_y:
        if x == 0:
            y_1.append([1, 0])
        else:
            y_1.append([0, 1])
    y_1 = np.vstack(y_1)
    return y_1
