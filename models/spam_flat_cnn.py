import os

import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.metrics import Precision, Recall
from numpy import array

from data.spam_flat import load_data
from models.cnn_model import build_model

PATH = os.path.dirname(os.path.abspath(__file__))


def save_model(model, fname):
    if not os.path.exists(os.path.join(PATH, "trained")):
        os.mkdir(os.path.join(PATH, "trained"))
    model_path = os.path.join(PATH, "trained", fname)
    model.save(model_path)


def load_model(fname):
    model_path = os.path.join(PATH, "trained", fname)
    return tf.keras.models.load_model(model_path)


def train_model(max_length=350, vocab_size=10000):
    model = build_model(max_length=max_length, vocab_size=vocab_size)
    X_ws_train, X_ws_test, trainX, trainY, testX, testY = load_data(max_length, vocab_size)
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_precision', mode='max', verbose=1, save_best_only=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=[Precision(name="precision", thresholds=0.7),
                           Recall(name="recall")])
    history = model.fit([trainX, trainX, trainX], array(trainY), validation_data=([testX, testX, testX], testY),
                        epochs=50, verbose=2,
                        callbacks=[EarlyStopping("val_precision", patience=10, restore_best_weights=True), checkpoint])
    return model, history


if __name__ == "__main__":
    model, history = train_model()
    print(model.summary())
    save_model(model, "spam_flat_cnn.h5")
