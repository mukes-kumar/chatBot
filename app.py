import json
import pickle
import random
import re
import uuid

import nltk
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

frontend_url = "https://main.d1kf7sgsxie4fi.amplifyapp.com/"


app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": [frontend_url, "http://localhost:3000"]}})

# --- Load all the trained model files ---
lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("chatbot_model.h5")

# --- In-memory "notebook" for storing user data during a conversation ---
session_data = {}


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


def predict_class(sentence, session_id):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []

    context = session_data.get(session_id, {}).get("context", None)

    # If context is set, we prioritize the intents that match that context
    if context:
        for intent in intents["intents"]:
            if intent.get("context_filter") == context:
                # For capture intents with no patterns, we assume it's the right intent
                return_list.append({"intent": intent["tag"], "probability": "1.0"})
        return return_list  # Return immediately with the context-specific intent

    # If no context, use the ML model to predict
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})

    return return_list


def get_response(intents_list, intents_json, session_id, user_message):
    if not intents_list:
        return "I'm sorry, I don't quite understand. You can ask me to 'book a service' or ask about our services."

    tag = intents_list[0]["intent"]

    if session_id not in session_data:
        session_data[session_id] = {}

    result = "Something went wrong. Please try again."

    for i in intents_json["intents"]:
        if i["tag"] == tag:
            # --- ENTITY EXTRACTION & SAVING TO SESSION ---
            context = session_data.get(session_id, {}).get("context", None)

            if context == "awaiting_name":
                session_data[session_id]["name"] = user_message.strip().title()
            if context == "awaiting_email":
                session_data[session_id]["email"] = user_message.strip().then()

            if context == "awaiting_phone":
                phone_match = re.search(r"\b\d{10}\b", user_message)
                phone = phone_match.group(0) if phone_match else "Not Provided"
                session_data[session_id]["phone"] = phone

            if context == "awaiting_address":
                session_data[session_id]["address"] = user_message.strip()

            if context == "awaiting_device":
                session_data[session_id]["device"] = user_message.strip()

            if context == "awaiting_problem":
                session_data[session_id]["problem"] = user_message.strip()

            # --- CONTEXT MANAGEMENT ---
            if "context_set" in i:
                session_data[session_id]["context"] = i["context_set"]
                # If we are restarting the form, clear old data
                if i["context_set"] == "awaiting_name":
                    session_data[session_id].pop("phone", None)
                    session_data[session_id].pop("address", None)
                    session_data[session_id].pop("device", None)
                    session_data[session_id].pop("problem", None)
            else:
                # If an intent has no context_set, it ends the flow. Clear context.
                session_data[session_id].pop("context", None)

            # --- DYNAMIC RESPONSE FORMATTING ---
            result = random.choice(i["responses"])
            user_info = session_data.get(session_id, {})
            for key, value in user_info.items():
                result = result.replace(f"{{{key}}}", str(value))

            break

    return result


# --- NEW: Add a Welcome Route for the Browser ---
@app.route("/", methods=["GET"])
def welcome():
    # We will return a beautiful HTML response
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Fix Mantra Chatbot API</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                background-color: #f0f2f5;
                color: #333;
            }}
            .container {{
                text-align: center;
                padding: 40px;
                background-color: white;
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            }}
            h1 {{
                color: #1a73e8;
                font-size: 2.5em;
                margin-bottom: 10px;
            }}
            p {{
                font-size: 1.2em;
                margin-top: 0;
            }}
            .status {{
                display: inline-flex;
                align-items: center;
                font-size: 1.1em;
                background-color: #e6f4ea;
                color: #34a853;
                padding: 10px 20px;
                border-radius: 20px;
                font-weight: 500;
            }}
            .status-dot {{
                height: 12px;
                width: 12px;
                background-color: #34a853;
                border-radius: 50%;
                display: inline-block;
                margin-right: 10px;
                animation: pulse 2s infinite;
            }}
            @keyframes pulse {{
                0% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(52, 168, 83, 0.7); }}
                70% {{ transform: scale(1); box-shadow: 0 0 0 10px rgba(52, 168, 83, 0); }}
                100% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(52, 168, 83, 0); }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Fix Mantra Chatbot API</h1>
            <p>The AI brain is up and running!</p>
            <div class="status">
                <span class="status-dot"></span>
                <span>Server is Online</span>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content


@app.route("/predict", methods=["POST"])
def predict():
    message = request.json.get("message")
    session_id = request.json.get("session_id")

    if not message or not session_id:
        return jsonify({"error": "Missing message or session_id"}), 400

    ints = predict_class(message, session_id)
    res = get_response(ints, intents, session_id, message)

    return jsonify({"reply": res})


# if __name__ == '__main__':
# app.run(debug=True)

if __name__ == "__main__":
    from waitress import serve

    print("Server is running on http://localhost:8080")
    serve(app, host="0.0.0.0", port=8080)
