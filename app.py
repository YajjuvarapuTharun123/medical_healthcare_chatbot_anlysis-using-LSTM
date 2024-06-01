from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import json

app = Flask(__name__)

# Load the model
model = load_model('medical_healthcare_chatbot_model.h5')

# Load tokenizer
with open('healthcare_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load label encoder
with open('healthcare_label_encoder.pickle', 'rb') as handle:
    lbl_enc = pickle.load(handle)

# Load intents file
with open('intents.json', 'r') as f:
    intents = json.load(f)

def get_response(intent):
    for i in intents['intents']:
        if i['intent'] == intent:
            return np.random.choice(i['responses'])

def predict_class(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=20, padding='post')
    pred = model.predict(padded)
    return lbl_enc.inverse_transform([pred.argmax()])[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.json['message']
    intent = predict_class(message)
    response = get_response(intent)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)