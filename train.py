print("STARTING TRAINING...")
import nltk
nltk.download('punkt_tab')
import json
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

lemmatizer = WordNetLemmatizer()

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, 'intents.json')

with open(r"C:\Users\sreeh\OneDrive\Desktop\INTERNSHIP PROJECT\chatbot_project\intents.json") as f:
    data = json.load(f)

words = []
labels = []
docs = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        docs.append((tokens, intent['tag']))
    labels.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w.isalpha()]
words = sorted(set(words))

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

def bag_of_words(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(w.lower()) for w in sentence_words]

    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

X = []
y = []

for doc in docs:
    X.append(bag_of_words(" ".join(doc[0])))
    y.append(label_encoder.transform([doc[1]])[0])

X = np.array(X)
y = np.array(y)

model = Sequential()
model.add(Dense(128, input_shape=(len(X[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(labels), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y, epochs=200, batch_size=8)

model.save("intent_model.h5")

print("Model trained and saved!")