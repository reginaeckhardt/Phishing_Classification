import string

import spacy
from alibi.explainers import AnchorText
from alibi.utils.download import spacy_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot

from spam_flat import load_data
from spam_flat import stemSentence
from models.spam_flat_cnn import load_model

VOCAB_SIZE = 10000
MAX_LENGTH = 350

class_names = ['Spam', 'Phishing']

print("Loading data...")
X_ws_train, X_ws_test, trainX, trainY, testX, testY = load_data(MAX_LENGTH, VOCAB_SIZE)

print("Loading model...")
model = load_model("spam_flat_cnn.h5")

print("Loading NLP model...")
model_en = 'en_core_web_md'
spacy_model(model=model_en)
nlp = spacy.load(model_en)


# # we have to define a predict function that includes the transformation of the list of texts to a vector of numerical values
def predict_cnn(text):
    stem_texts = (stemSentence(x) for x in text)
    encoded_texts = [one_hot(t, VOCAB_SIZE) for t in stem_texts]
    encoded_input = pad_sequences(encoded_texts, maxlen=MAX_LENGTH, padding="post", truncating="post")
    return model.predict([encoded_input, encoded_input, encoded_input])


print("Initialize Anchor")
explainer_anchor_cnn = AnchorText(nlp, predict_cnn)

print("Remove all non-standard characters from e-mails")
legal_characters = string.digits + string.ascii_letters + string.punctuation + ' '
X_ws_test1 = ["".join(x if x in legal_characters else "" for x in line) for line in X_ws_test]

assert len(X_ws_test1) == len(X_ws_test)

for idx, test_email in enumerate(X_ws_test1[:50]):
    if testY[idx][1] == 1:
        print(test_email)
        print('Position Phishing E-Mail in Test Data', X_ws_test1.index(test_email))
        explanation_anchor_cnn = explainer_anchor_cnn.explain(test_email, desired_label=1, threshold=0.75,
                                                              use_similarity_proba=False,
                                                              use_unk=True, sample_proba=0.5, beam_size=1, tau=0.15)
        print('Anchor: %s' % (' AND '.join(explanation_anchor_cnn['names'])))
        print('Precision: %.2f' % explanation_anchor_cnn['precision'])
        print(explanation_anchor_cnn)
