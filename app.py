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


# ... (all your existing imports: json, pickle, Flask, etc.)
# ... (your CORS setup, model loading, and all other functions remain the same)


# --- NEW, UPGRADED WELCOME & CHAT UI ROUTE ---
@app.route("/", methods=["GET"])
def welcome():
    # This HTML now includes a welcome screen, the chat UI, and advanced CSS/JS for animations and responsiveness.
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Flex Mitra AI Assistant</title>
        <style>
            /* --- General Styles --- */
            :root {{
                --primary-color: #1a73e8;
                --user-message-color: #1a73e8;
                --bot-message-color: #e9e9eb;
                --background-color: #f0f2f5;
            }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                background-color: var(--background-color);
                overflow: hidden; /* Prevents scrollbars on the body */
            }}

            /* --- Welcome Screen Styles --- */
            .welcome-screen {{
                text-align: center;
                transition: opacity 0.5s ease-out;
            }}
            .welcome-screen.hidden {{
                opacity: 0;
                pointer-events: none;
            }}
            .welcome-logo {{
                width: 100px;
                height: 100px;
                background-color: var(--primary-color);
                border-radius: 50%;
                display: flex;
                justify-content: center;
                align-items: center;
                color: white;
                font-size: 48px;
                font-weight: bold;
                animation: pulse 2s infinite;
                margin: 0 auto 20px auto;
            }}
            .welcome-screen h1 {{
                color: #333;
                font-size: 1.8em;
            }}

            /* --- Chat UI Styles --- */
            .chat-container {{
                width: 480px;
                height: 550px;
                background-color: white;
                border-radius: 15px;
                box-shadow: 0 5px 25px rgba(0, 0, 0, 0.15);
                display: flex;
                flex-direction: column;
                overflow: hidden;
                position: absolute; /* Allows it to be layered on top */
                transition: opacity 0.5s ease-in, transform 0.5s ease-in;
            }}
            .chat-container.hidden {{
                opacity: 0;
                transform: scale(0.95);
                pointer-events: none;
            }}
            .chat-header {{
                background-color: var(--primary-color);
                color: white;
                padding: 15px;
                text-align: center;
                font-size: 1.2em;
                font-weight: 500;
                flex-shrink: 0;
            }}
            .message-area {{
                flex-grow: 1;
                padding: 15px;
                overflow-y: auto;
                display: flex;
                flex-direction: column;
            }}
            .chat-message {{
                padding: 10px 15px;
                border-radius: 18px;
                margin-bottom: 10px;
                max-width: 80%;
                word-wrap: break-word;
                animation: slideUpFadeIn 0.4s ease-out;
            }}
            .user-message {{
                background-color: var(--user-message-color);
                color: white;
                align-self: flex-end;
            }}
            .bot-message {{
                background-color: var(--bot-message-color);
                color: #333;
                align-self: flex-start;
            }}
            .chat-input-form {{
                display: flex;
                padding: 10px;
                border-top: 1px solid #ddd;
                background-color: #fff;
                flex-shrink: 0;
            }}
            .chat-input-form input {{
                flex-grow: 1; border: 1px solid #ccc; border-radius: 20px;
                padding: 10px 15px; font-size: 1em; margin-right: 10px;
            }}
            .chat-input-form button {{
                background-color: var(--primary-color); color: white; border: none;
                border-radius: 20px; padding: 10px 20px; font-size: 1em; cursor: pointer;
            }}
            .typing-indicator {{
                align-self: flex-start; display: flex; align-items: center; padding: 10px 15px;
            }}
            .typing-indicator span {{
                height: 8px; width: 8px; margin: 0 2px;
                background-color: #a0a0a0; border-radius: 50%;
                display: inline-block; animation: typing-pulse 1.4s infinite ease-in-out both;
            }}
            .typing-indicator span:nth-child(1) {{ animation-delay: -0.32s; }}
            .typing-indicator span:nth-child(2) {{ animation-delay: -0.16s; }}
            
            /* --- Animations --- */
            @keyframes pulse {{
                0% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(26, 115, 232, 0.7); }}
                70% {{ transform: scale(1); box-shadow: 0 0 0 10px rgba(26, 115, 232, 0); }}
                100% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(26, 115, 232, 0); }}
            }}
            @keyframes slideUpFadeIn {{
                from {{ opacity: 0; transform: translateY(10px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            @keyframes typing-pulse {{ 0%, 80%, 100% {{ transform: scale(0); }} 40% {{ transform: scale(1.0); }} }}

            /* --- Responsive Design for Mobile --- */
            @media (max-width: 500px) {{
                .chat-container {{
                    width: 100vw;
                    height: 100vh;
                    border-radius: 0;
                    box-shadow: none;
                }}
            }}
        </style>
    </head>
    <body>
        <!-- Welcome Screen -->
        <div class="welcome-screen" id="welcome-screen">
            <div class="welcome-logo">FM</div>
            <h1>Welcome to Flex Mitra</h1>
            <p>Your Personal AI Assistant</p>
        </div>

        <!-- Chat UI (Initially Hidden) -->
        <div class="chat-container hidden" id="chat-container">
            <div class="chat-header">Fix Mintra AI Assistant</div>
            <div class="message-area" id="message-area">
                <!-- Initial bot message will be added by script -->
            </div>
            <form class="chat-input-form" id="chat-form">
                <input type="text" id="user-input" placeholder="Type a message..." autocomplete="off">
                <button type="submit">Send</button>
            </form>
        </div>

        <script>
            document.addEventListener('DOMContentLoaded', () => {{
                const welcomeScreen = document.getElementById('welcome-screen');
                const chatContainer = document.getElementById('chat-container');
                const chatForm = document.getElementById('chat-form');
                const userInput = document.getElementById('user-input');
                const messageArea = document.getElementById('message-area');
                
                const sessionId = Date.now().toString() + Math.random().toString();

                // --- Transition from Welcome to Chat ---
                setTimeout(() => {{
                    welcomeScreen.classList.add('hidden');
                    chatContainer.classList.remove('hidden');
                    addMessage('bot', "Hello! I'm the AI assistant for Fix Mintra , Welcome to Fix Mantra. How can I help you with your appliance or gadget repair today?"");
                }}, 4000); // 4-second delay

                // --- Chat Logic ---
                chatForm.addEventListener('submit', async (event) => {{
                    event.preventDefault();
                    const userMessage = userInput.value.trim();
                    if (!userMessage) return;

                    addMessage('user', userMessage);
                    userInput.value = '';
                    showTypingIndicator();

                    try {{
                        const response = await fetch('/predict', {{
                            method: 'POST',
                            headers: {{ 'Content-Type': 'application/json' }},
                            body: JSON.stringify({{ message: userMessage, session_id: sessionId }})
                        }});
                        
                        const data = await response.json();
                        const botReply = data.reply || "Sorry, I encountered an error.";
                        
                        hideTypingIndicator();
                        addMessage('bot', botReply);

                    }} catch (error) {{
                        console.error('Error:', error);
                        hideTypingIndicator();
                        addMessage('bot', "I can't connect to my brain right now. Please try again later.");
                    }}
                }});

                function addMessage(sender, text) {{
                    const messageDiv = document.createElement('div');
                    messageDiv.classList.add('chat-message', sender === 'user' ? 'user-message' : 'bot-message');
                    messageDiv.innerText = text;
                    messageArea.appendChild(messageDiv);
                    messageArea.scrollTop = messageArea.scrollHeight;
                }}

                function showTypingIndicator() {{
                    const typingDiv = document.createElement('div');
                    typingDiv.id = 'typing-indicator';
                    typingDiv.classList.add('chat-message', 'typing-indicator');
                    typingDiv.innerHTML = '<span></span><span></span><span></span>';
                    messageArea.appendChild(typingDiv);
                    messageArea.scrollTop = messageArea.scrollHeight;
                }}

                function hideTypingIndicator() {{
                    const typingIndicator = document.getElementById('typing-indicator');
                    if (typingIndicator) {{
                        typingIndicator.remove();
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """
    return html_content


# ... (your @app.route('/predict') and other functions follow here)


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
