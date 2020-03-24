#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import pickle

import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

PATH = os.path.dirname(os.path.abspath(__file__))

porter = PorterStemmer()


def preprocess_data(max_length, vocab_size):
    if not os.path.exists(os.path.join(PATH, "raw", "spam-flat3.csv")):
        raise FileNotFoundError("Please place 'spam-flat3.csv in folder data/raw/")

    path = os.path.join(PATH, "raw", "spam-flat3.csv")
    mydata = pd.read_csv(path)
    # with_subject = mydata["subject"] + " " + mydata["bodyHtml"]
    # with_subject = with_subject.fillna("Ignore")
    texts = mydata["bodyHtml"]
    texts = texts.fillna("Ignore")
    # X_ws = with_subject.tolist()
    X_ws = texts.tolist()
    mydata.loc[mydata['phishing'].astype(str) == 'False', 'phishing'] = 0
    mydata.loc[mydata['phishing'].astype(str) == 'True', 'phishing'] = 1
    y = mydata['phishing'].astype(int).tolist()
    X_ws_train, X_ws_test, y_train, y_test = train_test_split(X_ws, y, test_size=0.3, random_state=42)
    trainX = preprocess_text(X_ws_train, max_length=max_length, vocab_size=vocab_size)
    testX = preprocess_text(X_ws_test, max_length=max_length, vocab_size=vocab_size)
    trainY = preprocess_label(y_train)
    testY = preprocess_label(y_test)
    data = (X_ws_train, X_ws_test, trainX, trainY, testX, testY)
    if not os.path.exists(os.path.join(PATH, "processed")):
        os.mkdir(os.path.join(PATH, "processed"))

    with open(os.path.join(PATH, "processed", f"spam-flat3_{max_length}_{vocab_size}.pickle"), "wb") as f:
        pickle.dump(data, f)


def load_data(max_length, vocab_size):
    if not os.path.exists(os.path.join(PATH, "processed", f"spam-flat3_{max_length}_{vocab_size}.pickle")):
        print("Data not found, preprocessing data...")
        preprocess_data(max_length, vocab_size)
        print("Done preprocessing")

    with open(os.path.join(PATH, "processed", f"spam-flat3_{max_length}_{vocab_size}.pickle"), "rb") as f:
        return pickle.load(f)


def stemSentence(sentence):
    token_words = word_tokenize(sentence)
    stem_sentence = (porter.stem(word) for word in token_words)
    return " ".join(stem_sentence)


def preprocess_text(data_x, max_length=350, vocab_size=10000):
    X_2 = (stemSentence(x) for x in data_x)
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


if __name__ == "__main__":
    load_data(350, 10000)
