import os

from data.spam_flat import load_data
from models.cnn import build_model

PATH = os.path.dirname(os.path.abspath(__file__))


def load_model():
    model_path = os.path.join(PATH, "trained", "model_name.hdf5")

    return model


def train_model():
    data = load_data(vocab_size)
    model = build_model(vocab_size)

    model.compile()
    model.fit()

    # save model


if __name__ == "__main__":
    train_model()
