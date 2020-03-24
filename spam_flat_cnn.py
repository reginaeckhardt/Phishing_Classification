import os
import numpy as np
import tensorflow as tf
from numpy import array
from keras import metrics
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.metrics import Precision, Recall
from cnn_model import build_model
from spam_flat import load_data

PATH = os.path.dirname(os.path.abspath("spam-flat3.csv"))


# In[6]:
def save_model(model, fname):
    model_path = os.path.join(PATH, fname)
    model.save(model_path)

def load_model():
    model_path = os.path.join(PATH, "model.h5")
    return tf.keras.models.load_model(model_path)


# In[7]:


def train_model(max_length=350, vocab_size = 10000):
    #data = load_data(vocab_size)
    model = build_model(max_length=max_length, vocab_size = vocab_size)
    X_ws_train, X_ws_test, trainX, trainY, testX, testY = load_data(max_length, vocab_size)
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_precision', mode='max', verbose=1, save_best_only=True) 
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[Precision(name="precision", thresholds = 0.7),
	Recall(name="recall")])
    history = model.fit([trainX,trainX,trainX], array(trainY), validation_data=([testX,testX,testX],testY), epochs=50, verbose=2,callbacks=[EarlyStopping("val_precision", patience=10, restore_best_weights= True),checkpoint])
#    model.save('model.h5')
#    print(history_cnn.summary())
 #   print(model.summary())
    return model, history


# In[8]:


if __name__ == "__main__":
    model, history = train_model()
    print(model.summary())
    save_model(model,"spam_flat_cnn.h5")

# In[3]:


# from numpy import array
# from keras.callbacks import EarlyStopping
# from keras.callbacks import ModelCheckpoint
# vocab_size = 10000
# data = load_data(vocab_size)
# model = build_model(vocab_size)
# X_ws_train, X_ws_test, trainX, trainY, testX, testY = load_data(vocab_size)


# # In[7]:


# import numpy as np
# np.count_nonzero(trainY == 1)


# # In[8]:


# len(trainY)


# # In[9]:


# #import os
# PATH = os.path.dirname(os.path.abspath("spam-flat3.csv"))
# import pandas as pd
# from sklearn.model_selection import train_test_split
# path = os.path.join(PATH, "spam-flat3.csv")
# mydata = pd.read_csv(path)
# with_subject = mydata["subject"] + " " + mydata["bodyHtml"]
# with_subject = with_subject.fillna("Ignore")
# X_ws = with_subject.tolist()
# mydata.loc[mydata['phishing'].astype(str) == 'False','phishing'] = 0
# mydata.loc[mydata['phishing'].astype(str) == 'True','phishing'] = 1
# y = mydata['phishing'].astype(int).tolist()
# X_ws_train,X_ws_test,y_train,y_test = train_test_split(X_ws,y,test_size=0.3,random_state=42)


# # In[16]:


# #np.count_nonzero(y_test == 1)


# # In[17]:


# #import numpy as np
# #unique, counts = np.unique(y_test, return_counts=True)
# #dict(zip(unique, counts))


# # In[ ]:




