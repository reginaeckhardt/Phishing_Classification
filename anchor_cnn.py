#import itertools
import os
import string
import spacy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from alibi.explainers import AnchorText
from alibi.utils.download import spacy_model
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
    
from data.spam_flat import load_data
from data.spam_flat import stemSentence
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

def compute_anchor(text):
    print("Initialize Anchor")
    explainer_anchor_cnn = AnchorText(nlp, predict_cnn)
    explanation_anchor_cnn = explainer_anchor_cnn.explain(text, desired_label=1, threshold=0.75,
                                                              use_similarity_proba=False, use_unk=True, sample_proba=0.5, beam_size=1, tau=0.15)
    del explainer_anchor_cnn
    return explanation_anchor_cnn
                                                          
print("Remove all non-standard characters from e-mails")
legal_characters = string.digits + string.ascii_letters + string.punctuation + ' '
X_ws_test1 = ["".join(x if x in legal_characters else "" for x in line) for line in X_ws_test]

assert len(X_ws_test1) == len(X_ws_test)


for idx, test_email in enumerate(X_ws_test1[30:40]):
    if testY[idx][1] == 1:
        print(test_email)
        print('Position Phishing E-Mail in Test Data', X_ws_test1.index(test_email))
        explanation_anchor_cnn = compute_anchor(test_email)
        #explanation_anchor_cnn = explainer_anchor_cnn.explain(test_email, desired_label=1, threshold=0.75,
                                                              #use_similarity_proba=False,
                                                              #use_unk=True, sample_proba=0.5, beam_size=1, tau=0.15)
        print('Anchor: %s' % (' AND '.join(explanation_anchor_cnn['names'])))
        print('Precision: %.2f' % explanation_anchor_cnn['precision'])
        print(explanation_anchor_cnn)

# for i in range(len(X_ws_test)):
    # if int(round(predict_cnn([X_ws_test[i]])[0,1],0))==1 and testY[i][1]==1:
        
        # print('E-Mail:',i)
# #test_email = X_ws_test[idx]
        # pred = class_names[int(round(predict_cnn([X_ws_test[i]])[0,1],0))]
        # alternative =  class_names[int(round(1 - predict_cnn([X_ws_test[i]])[0,0]))]
        # print('Prediction: %s' % pred)
        # print('True Class:', class_names[testY[i][1]])
        # print('Probability(Phishing) =', predict_cnn([X_ws_test[i]])[0,1])

# Remove special signs 
#words = ['©','@','/',':','邀请您观看','的相册','年2月3日','提供者','查看相册','播放幻灯片','来自',
#         '的消息','如果您在阅读此电子邮件时出现问题，请将以下地址复制并粘贴到您的浏览器中',
#         '要分享您的照片或在朋友与您分享照片时收到通知，','请获取属于您自己的免费','%'
#         ,'网络相册帐户。','–','’','：','»','•','®','™','пїЅ','�']

#X_ws_test1 = [] 
#for line in X_ws_test[1200:1500]:
#    for w in words:
#        line = line.replace(w, '')
#    X_ws_test1.append(line)
#
#for test_email in X_ws_test1:
#    if  testY[X_ws_test1.index(test_email)][1]== 1:
#        print(test_email)
#        print('Position Phishing E-Mail in Test Data', X_ws_test1.index(test_email))
#        for threshold, sample_proba, beam_size, tau in itertools.product(thresholds, sample_probas, beam_sizes, taus):
#            explanation_anchor_cnn = explainer_anchor_cnn.explain(test_email, threshold=threshold, use_similarity_proba=False,
#                                                                  use_unk=True, sample_proba=sample_proba, beam_size=beam_size, tau=tau)
#        explanation_anchor_cnn = explainer_anchor_cnn.explain(test_email, threshold=0.95,use_similarity_proba=False,
#                               use_unk=True, sample_proba=0.5,beam_size=5, tau = 0.7)
#        print('Anchor: %s' % (' AND '.join(explanation_anchor_cnn['names'])))
#        print('Precision: %.2f' % explanation_anchor_cnn['precision'])
#        print(explanation_anchor_cnn)



# print('Anchor: %s' % (' AND '.join(explanation_anchor_cnn['names'])))
# print('Precision: %.2f' % explanation_anchor_cnn['precision'])
# print('\nExamples where anchor applies and model predicts %s:' % pred)
# print('\n'.join([x[0] for x in explanation_anchor_cnn['raw']['examples'][-1]['covered_true']]))
# print('\nExamples where anchor applies and model predicts %s:' % alternative)
# print('\n'.join([x[0] for x in explanation_anchor_cnn['raw']['examples'][-1]['covered_false']]))


# # In[15]:


#print(explanation_anchor_cnn)


# # In[210]:


#print(test_email)
#X_ws_test_1 = X_ws_test[535:537]
#filtered_numbers = [number for number in test_email if number != '©']
#print(X_ws_test_1)
#words = ['©']
## # In[211]:
#item1 = [] 
#for line in X_ws_test_1:
#    for w in words:
#        line = line.replace(w, '')
#    item1.append(line)
#print(item1)
# print(explanation_anchor_cnn)


# # In[1]:


# pred_cnn = class_names[int(round(predict_cnn([test_email])[0,1]))]
# alternative_cnn =  class_names[1 - int(round(predict_cnn([test_email])[0,1]))]
# print('Probability(Phishing) in the CNN =', predict_cnn([test_email])[0,1])
# print('Prediction CNN: %s' % pred_cnn)
# print('True class: %s' % class_names[y_test[idx]])


# # In[ ]:




