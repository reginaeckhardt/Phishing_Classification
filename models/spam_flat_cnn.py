import os

import tensorflow as tf

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import numpy as np

from models.cnn_model import build_model
from data.spam_flat import load_data

PATH = os.path.dirname(os.path.abspath(__file__))


def save_model(model, fname):
    model_path = os.path.join(PATH, "trained", fname)
    model.save(model_path)


def load_model(fname):
    model_path = os.path.join(PATH, "trained", fname)
    return tf.keras.models.load_model(model_path)


def train_model(vocab_size=10000):
    model = build_model(vocab_size)
    X_ws_train, X_ws_test, trainX, trainY, testX, testY = load_data(vocab_size)

    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit([trainX, trainX, trainX], np.array(trainY), validation_data=([testX, testX, testX], testY),
                        epochs=50, batch_size=16, verbose=2, callbacks=[EarlyStopping(patience=3), checkpoint])

    return model, history


if __name__ == "__main__":
    model, history = train_model()
    print(model.summary())
    model.save('model.h5')
