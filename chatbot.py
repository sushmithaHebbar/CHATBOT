import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import pickle
import re
import requests
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()
# IMPORTANT: Replace with your actual Google API key.
# This key is a placeholder and will not work.
GOOGLE_API_KEY = os.getenv("API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("API_KEY is not found")
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"

# Initialize the lemmatizer

lemmatizer = WordNetLemmatizer()
# --- Load and Preprocess Data ---

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

# Save the lists for later use

with open('words.pkl', 'wb') as f:
    pickle.dump(words, f)
with open('classes.pkl', 'wb') as f:
    pickle.dump(classes, f)

# --- Create the Training Data ---

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

# --- Train the Machine Learning Model ---
clf = SVC(kernel='linear', probability=True)
clf.fit(train_x, np.argmax(train_y, axis=1))

with open('chatbot_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

# --- Prediction and Response Generation ---

def clean_up_sentence(sentence):
    """Tokenize and lemmatize a user's input sentence."""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence, words, show_details=False):
    """Create a bag of words from a sentence."""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    """Predict the class (intent) of a sentence."""
    with open('chatbot_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('words.pkl', 'rb') as f:
        words = pickle.load(f)
    with open('classes.pkl', 'rb') as f:
        classes = pickle.load(f)

    p = bag_of_words(sentence, words)
    res = model.predict_proba([p])[0]
    # Get the index of the highest probability
    max_prob_index = np.argmax(res)
    # Get the class name and confidence score
    predicted_class = classes[max_prob_index]
    confidence = res[max_prob_index]
    return predicted_class, confidence

def get_response(intent_tag, user_name=None):
    """Get a random response based on the predicted intent tag."""
    for intent in intents['intents']:
        if intent['tag'] == intent_tag:
            response = random.choice(intent['responses'])
            if user_name and "[name]" in response:
                return response.replace("[name]", user_name.capitalize())
            return response
    return "I'm sorry, I don't understand that."


def get_llm_response(prompt):
    """Sends a prompt to the Gemini API and returns the response and sources."""
    if not GOOGLE_API_KEY:
        return "I can't connect to my brain right now. Please check my API key.", []
    headers = {'Content-Type': 'application/json'}
    params = {'key': GOOGLE_API_KEY}
   
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "tools": [
            {
                "google_search": {}
            }
        ]
    }
   
    try:
        response = requests.post(API_URL, headers=headers, params=params, json=payload)
        response.raise_for_status() # Raise an error for bad status codes

        result = response.json()

        # Check if the response contains generated text
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

# A simple user state dictionary to store dynamic information
user_data = {"name": None}

def extract_name(text):
    """
    Attempts to extract a name from a user's input using a simple rule.
    """
    match = re.search(r'(my name is|i am|you can call me|im)\s+([A-Za-z]+)', text, re.IGNORECASE)
    if match:
        return match.group(2)
   
    words = text.split()
    if len(words) > 2:
        return words[-1]
    return None


def save_to_history(sender, message):
    """Appends a message to the history.txt file with a timestamp."""
    with open('history.txt', 'a') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {sender}: {message}\n")

# Main chatbot loop

print("Chatbot is live! Type 'quit' to exit.")

while True:
    user_input = input("You: ")
    save_to_history("You", user_input)
    if user_input.lower() == 'quit':
        save_to_history("Chatbot", "Goodbye!")
        break

    predicted_class, confidence = predict_class(user_input)
    # Check if the confidence is high enough to use the local intent

    if confidence > 0.8: # You can adjust this threshold
        if predicted_class == 'user_introduction':
            name = extract_name(user_input)
            if name:
                user_data["name"] = name
                response = get_response(predicted_class, user_name=name)
            else:
                response = "Hello! It's nice to meet you, but I didn't catch your name."
        else:
            response = get_response(predicted_class, user_name=user_data["name"])
        print(f"Chatbot: {response}")
        save_to_history("Chatbot", response)
    else:
        # If confidence is low, fall back to the LLM
        response, sources = get_llm_response(user_input)
       
        # Check if the user's input is a name, and if so, try to extract and store it
        if "name is" in user_input.lower():
            name = extract_name(user_input)
            if name:
                user_data["name"] = name
                response = f"Hello, {name}! It's nice to meet you. " + response

        print(f"Chatbot: {response}")
        save_to_history("Chatbot", response)
       
        if sources:
            source_text = "\nSources:\n"
            for source in sources:
                source_text += f"- {source['title']} ({source['uri']})\n"
            print(source_text)
            save_to_history("Chatbot", source_text)