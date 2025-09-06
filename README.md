# CHATBOT

AI Chatbot with Full-Stack Web Interface

This project is a sophisticated chatbot that combines a traditional machine learning model for intent recognition with a large language model (LLM) for general conversation capabilities. The web interface is built using Flask and HTML/CSS/JavaScript.

## Features

- Intent recognition using a trained ML model
- Large language model (LLM) fallback for broader conversation
- Easy-to-use web interface
- API key management via `.env` file

## Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/sushmithaHebbar/CHATBOT.git
   cd CHATBOT
   ```

2. **Set Up Your Environment**

   It is highly recommended to use a virtual environment to manage dependencies.

   ```bash
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**

   Install all the required Python libraries using the `requirements.txt` file.

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Your API Key**

   Create a `.env` file in the root directory of the project and add your Google Gemini API key to it.

   ```
   API_KEY="YOUR_API_KEY_HERE"
   ```

5. **Run the Server**

   Your `app.py` file contains both the model training and the Flask server. The first time you run it, it will train the model and save the necessary files (`chatbot_model.pkl`, `words.pkl`, `labels.pkl`).

   ```bash
   python app.py
   ```

   You should see output indicating that the Flask server is running on [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Usage

Open your web browser and navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000). You will see the chatbot interface.

Type your message and interact with the AI chatbot.

---
