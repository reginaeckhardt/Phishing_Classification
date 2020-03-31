# import itertools

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

from data.spam_flat import stemSentence
from models.spam_flat_cnn import load_model

VOCAB_SIZE = 10000
MAX_LENGTH = 350

model = load_model("spam_flat_cnn.h5")


def pad_text(encoded_text):
    if len(encoded_text) < MAX_LENGTH:
        return encoded_text + [0]*(MAX_LENGTH - len(encoded_text))
    else:
        return encoded_text[:MAX_LENGTH]


def predict_cnn(text):
    stem_texts = (stemSentence(x) for x in text)
    encoded_texts = (one_hot(t, VOCAB_SIZE) for t in stem_texts)

    encoded_input = [pad_text(encoded_text) for encoded_text in encoded_texts]

    return model.predict([encoded_input, encoded_input, encoded_input])


if __name__ == "__main__":
    print(predict_cnn(["Hello, please click this link!"]))
