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
from keras.models import load_model

import speech_recognition as sr
import pyttsx3

import random
import pickle
import json
import os


print("\n\nHello Human!!")
print("Please wait while we check for updates..\n\n")

dir_name = os.path.dirname(os.path.abspath(__file__))
train = False
with open('files/meta.pkl','rb') as f:
    old_modified_time = f.read()
    corpus_path = os.path.join(dir_name,"corpus.json")
    new_modified_time = os.path.getmtime(corpus_path)
    if(new_modified_time>float(old_modified_time)):
        train = True

#Retraining    
if train:
    print("\n\nUpdating..\n\n")
    import projectguidingbot        #importing this will execute the imported script.  This is not the best way to do it, but for the sake of time and simplicity, I've chosen this.

with open("files/documents.pkl","rb") as f:
    documents = pickle.load(f)
with open("files/vectorizer.pkl","rb") as f:
    vectorizer = pickle.load(f)
f = open('corpus.json').read()
corpus = json.loads(f)
with open("files/labelencoder.pkl","rb") as f:
    le = pickle.load(f)
model = load_model('files/model.h5')

designations = {
    1: "intern",
    2: "jr dev",
    3: "sr dev",
    4: "executive"
}

interactions = {
    1: False,
    2: True
}

classes = le.classes_
stop_words = stopwords.words('english')
stop_words.extend(["?","!","."])
stop_words.remove('how')
stop_words.remove('what')
stop_words.remove('up')
spell = Speller('en')
lemmatizer = WordNetLemmatizer()

def preprocessing_pipeline(data):
    data = data.strip().lower()
    words = word_tokenize(data)
    words = [word for word in words if word not in stop_words]
    words = [spell(word) for word in words]
    words = [lemmatizer.lemmatize(word) for word in words]
    sent = " ".join(words)
    sent = [sent]
    X_vectorized = vectorizer.transform(sent).toarray()
    return X_vectorized

designation = ""
speech = False

def walkthrough():
    
    for intent in corpus['intents']:
        if(intent['access']=="all" or intent['access']==designation):
            if(intent['tag']!="greeting" and intent['tag']!="bye" and intent['tag']!="walkthrough"):
                print(random.choice(intent['responses']))


def get_response(X_vectorized):
    y_pred = model.predict(X_vectorized)
    y_pred = np.argmax(y_pred)
    tag = classes[y_pred]
    for intent in corpus['intents']:
        if(tag==intent['tag']):
            if((designation in intent['access']) or (intent['access']=="all")):
                response = random.choice(intent['responses'])
    return tag,response

def speak(line):
    engine = pyttsx3.init()
    engine.say(line)
    engine.runAndWait()


print("\n\n")
print(
    '''
    Please enter your designation:
    1. Enter 1 for intern
    2. Enter 2 for Junior dev
    3. Enter 3 for Senior dev
    4. Enter 4 for Executive
    '''
)
choice = int(input())
designation = designations.get(choice)

print("\n\n")
print(
    '''
    How do you want to interact?
    1. Enter 1 for Text
    2. Enter 2 for Text + Speech
    '''
)
choice = int(input())
speech = interactions.get(choice)

if(speech):
    r = sr.Recognizer()

print("We can start talking now!")
run = True
while(run):

    if(speech):
        try: 

            with sr.Microphone() as source2: 
                print()
                print("Listening..")
                
                r.adjust_for_ambient_noise(source2, duration=0.2) 
                
                audio2 = r.listen(source2) 
                
                inp = r.recognize_google(audio2) 
                print(f"YOU>> {inp}")
              
        except sr.RequestError as e: 
            print("Could not request results; {0}".format(e)) 
            speech = False
          
        except sr.UnknownValueError: 
            print("unknown error occured") 
            speech = False
    else:
        print()
        inp = str(input("YOU>> "))
    
    sentences = sent_tokenize(inp)
    for sentence in sentences:
        X_vectorized = preprocessing_pipeline(sentence)
        tag, response = get_response(X_vectorized)
        print(f"BOT>> {response}")
        if(speech):
            speak(response)
        if(tag=="bye"):
            run = False
        if(tag=="walkthrough"):
            walkthrough()
