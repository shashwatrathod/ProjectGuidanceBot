# -*- coding: utf-8 -*-
"""ProjectGuidingBot.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XR5t8L4EIc52M_G4IkJvOjq93q7MbOff
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install autocorrect

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from autocorrect import Speller
from nltk.tokenize import  sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, InputLayer
from keras.optimizers import Adam

import random
import pickle
import json
import os

def preprocessing_pipeline(data):
  
  data = data.strip().lower()
  words = word_tokenize(data)
  words = [word for word in words if word not in stop_words]
  words = [spell(word) for word in words]
  words = [lemmatizer.lemmatize(word) for word in words]
  return words

f = open('corpus.json').read()
corpus = json.loads(f)

words = []
classes = []
documents = []


stop_words = stopwords.words('english')
stop_words.extend(["?","!","."])
stop_words.remove('how')
stop_words.remove('what')
stop_words.remove('up')
spell = Speller('en')
lemmatizer = WordNetLemmatizer()

for intent in corpus['intents']:
  ws = []
  for pattern in intent['patterns']:
    if(intent['tag'] not in classes):
      w = preprocessing_pipeline(pattern)
      ws.append(w)
      words.extend(w)
  for w in ws:
    documents.append((w,intent['tag']))
  classes.append(intent['tag'])

X = []
Y = []
pickle.dump(documents,open('files/documents.pkl','wb'))
for x,y in documents:
  X.append(x)
  Y.append(y)

for i in range(len(X)):
  li = X[i]
  X[i] = " ".join(li)
vectorizer = CountVectorizer()
vectorizer.fit(X)
pickle.dump(vectorizer,open('files/vectorizer.pkl','wb'))

X_vectorized = vectorizer.transform(X).toarray()
le = LabelEncoder()
Y_encoded = le.fit_transform(Y)
pickle.dump(le,open('files/labelencoder.pkl',"wb"))
for i in range(len(X_vectorized)):
  X_vectorized[i] = np.asarray((X_vectorized[i]))

model = Sequential()
model.add(InputLayer(input_shape=(np.shape(X_vectorized)[1],)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(le.classes_), activation='softmax'))
adam = Adam(learning_rate=0.01)
model.compile(loss='sparse_categorical_crossentropy',optimizer=adam,metrics=['accuracy'])

model.summary()

history = model.fit(X_vectorized,np.array(Y_encoded),epochs=150,batch_size=5)

model.save('files/model.h5',history)

with open("files/meta.pkl","w") as f:
  dir_name = os.path.dirname(os.path.abspath(__file__))
  corpus_path = os.path.join(dir_name,"corpus.json")
  t = os.path.getmtime(corpus_path)
  f.write(str(t))