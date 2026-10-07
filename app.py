orders = {
    "12345": "Your order is shipped 🚚",
    "67890": "Your order is out for delivery 📦",
    "11111": "Your order has been delivered ✅"
}
import json
import numpy as np
import nltk
from flask import Flask, render_template, request, jsonify
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

with open('intents.json') as f:
    data = json.load(f)

model = load_model("intent_model.h5")

words = []
labels = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
    labels.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w.isalpha()]
words = sorted(set(words))

label_encoder = LabelEncoder()
label_encoder.fit(labels)

def bag_of_words(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(w.lower()) for w in sentence_words]

    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def get_response(user_input):
    # Check if user entered order ID
    if user_input in orders:
        return orders[user_input]

    bow = bag_of_words(user_input)
    res = model.predict(np.array([bow]))[0]
    idx = np.argmax(res)
    confidence = res[idx]

    tag = label_encoder.inverse_transform([idx])[0]

    if confidence > 0.7:
        for intent in data['intents']:
            if intent['tag'] == tag:
                return np.random.choice(intent['responses'])
    else:
        return "I'm not sure I understand. Please rephrase."

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]
    response = get_response(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)