from flask import Flask, request, jsonify, render_template
import json
import os
import re
from datetime import datetime
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import pickle
import random
import requests
from sklearn.svm import SVC
from flask_cors import CORS
from dotenv import load_dotenv

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("API_KEY is not found")
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# --- Model Training and Data Loading ---

# This function will run once to train and save the model if files don't exist.
def initialize_chatbot_model():
    if os.path.exists('chatbot_model.pkl') and os.path.exists('words.pkl') and os.path.exists('classes.pkl'):
        print("Model files found. Skipping training.")
        return

    try:
        with open('intents.json', 'r') as file:
            intents = json.load(file)
    except FileNotFoundError:
        print("Error: 'intents.json' not found. Please create the file with the provided content.")
        exit()

    words = []
    classes = []
    documents = []

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            w = nltk.word_tokenize(pattern)
            words.extend(w)
            documents.append((w, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    words = [lemmatizer.lemmatize(w.lower()) for w in words if w.isalnum()]
    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))

    with open('words.pkl', 'wb') as f:
        pickle.dump(words, f)
    with open('classes.pkl', 'wb') as f:
        pickle.dump(classes, f)

    training = []
    output_empty = [0] * len(classes)

    for doc in documents:
        bag = []
        pattern_words = doc[0]
        pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training, dtype=object)

    train_x = list(training[:, 0])
    train_y = list(training[:, 1])

    clf = SVC(kernel='linear', probability=True)
    clf.fit(train_x, np.argmax(train_y, axis=1))

    with open('chatbot_model.pkl', 'wb') as f:
        pickle.dump(clf, f)

    print("Model training complete and files saved.")

# Run initialization on startup
initialize_chatbot_model()

# --- Load the Model and Data for the application ---
try:
    with open('intents.json', 'r') as file:
        intents = json.load(file)
    with open('chatbot_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('words.pkl', 'rb') as f:
        words = pickle.load(f)
    with open('classes.pkl', 'rb') as f:
        classes = pickle.load(f)
except FileNotFoundError:
    print("Error: Required model files not found. Please ensure 'intents.json' exists and the model is trained.")
    exit()

# A simple user state dictionary to store dynamic information
user_data = {"name": None}

# --- Helper Functions ---
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    with open('intents.json', 'r') as file:
        intents = json.load(file)
    with open('chatbot_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('words.pkl', 'rb') as f:
        words = pickle.load(f)
    with open('classes.pkl', 'rb') as f:
        classes = pickle.load(f)
        
    p = bag_of_words(sentence, words)
    res = model.predict_proba([p])[0]
    max_prob_index = np.argmax(res)
    predicted_class = classes[max_prob_index]
    confidence = res[max_prob_index]
    return predicted_class, confidence

def get_response(intent_tag, user_name=None):
    for intent in intents['intents']:
        if intent['tag'] == intent_tag:
            response = random.choice(intent['responses'])
            if user_name and "[name]" in response:
                return response.replace("[name]", user_name.capitalize())
            return response
    return "I'm sorry, I don't understand that."

def get_llm_response(prompt):
    if not GOOGLE_API_KEY:
        return "I can't connect to my brain right now. Please check my API key.", []
    
    headers = {'Content-Type': 'application/json'}
    params = {'key': GOOGLE_API_KEY}
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}]
    }
    
    try:
        response = requests.post(API_URL, headers=headers, params=params, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # text = "I couldn't generate a response for that. Please try again."
        # sources = []
        
        # if 'candidates' in result and result['candidates']:
        #     parts = result['candidates'][0]['content']['parts']
        #     if parts:
        #         text = parts[0]['text']
        if 'candidates' in result and result['candidates']:
            text = result['candidates'][0]['content']['parts'][0]['text']

            # Extract and format sources from grounding metadata
            sources = []
            
            if 'groundingMetadata' in result['candidates'][0] and 'groundingAttributions' in result['candidates'][0]['groundingMetadata']:
                for attribution in result['candidates'][0]['groundingMetadata']['groundingAttributions']:
                    if 'web' in attribution:
                        sources.append({
                            'uri': attribution['web']['uri'],
                            'title': attribution['web']['title']
                        })
            return text, sources
        else:
            return "I couldn't generate a response for that. Please try again.", []

    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API: {e}")
        return "I'm having trouble connecting to my knowledge base. Please try again later.", []
    except KeyError:
        return "An unexpected error occurred with the API response.", []
user_data = {"name": None}

def extract_name(text):
    match = re.search(r'(my name is|i am|you can call me|im)\s+([A-Za-z]+)', text, re.IGNORECASE)
    if match:
        return match.group(2)
    
    words = text.split()
    if len(words) > 2:
        return words[-1]
    return None

def save_to_history(sender, message):
    with open('history.txt', 'a') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {sender}: {message}\n")

# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    save_to_history("You", user_input)

    predicted_class, confidence = predict_class(user_input)

    if confidence > 0.8:
        if predicted_class == 'user_introduction':
            name = extract_name(user_input)
            if name:
                user_data["name"] = name
                response = get_response(predicted_class, user_name=name)
            else:
                response = "Hello! It's nice to meet you, but I didn't catch your name."
        else:
            response = get_response(predicted_class, user_name=user_data["name"])
        
        save_to_history("Chatbot", response)
        return jsonify({'response': response, 'sources': None})
    else:
        response, sources = get_llm_response(user_input)
        
        name = extract_name(user_input)
        if name:
            user_data["name"] = name
            response = f"Hello, {name}! It's nice to meet you. " + response
        
        save_to_history("Chatbot", response)
        
        # if sources:
        #     source_text = "\nSources:\n"
        #     for source in sources:
        #         source_text += f"- [{source['title']}]({source['uri']})\n"
        #     save_to_history("Chatbot", source_text)
            
        return jsonify({'response': response, 'sources': sources})

@app.route('/history')
def get_history_titles():
    try:
        history = []
        if os.path.exists('history.txt'):
            with open('history.txt', 'r') as f:
                lines = f.readlines()
                if lines:
                    first_line = lines[0].strip()
                    history.append({'title': 'Current Chat', 'filename': 'history.txt'})
        return jsonify(history)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history/<filename>')
def get_chat_messages(filename):
    try:
        if filename == 'history.txt':
            messages = []
            with open('history.txt', 'r') as f:
                for line in f:
                    match = re.match(r'\[(.*?)\] (You|Chatbot): (.*)', line)
                    if match:
                        sender = match.group(2)
                        message = match.group(3)
                        messages.append({'sender': sender, 'message': message})
            return jsonify(messages)
        else:
            return jsonify({'error': 'Chat not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists('history.txt'):
        with open('history.txt', 'w') as f:
            pass
    app.run(debug=True)


