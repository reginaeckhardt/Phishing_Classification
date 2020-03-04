import os

PATH = os.path.dirname(os.path.abspath(__file__))


def load_data(vocab_size):
    path = os.path.join(PATH, "raw", "spam-flat3.csv")

    trainX = preprocess(trainX)

    return trainX, trainY, testX, testY


def preprocess(data_x):

    # stemming etc.

    return preprocessed_data

