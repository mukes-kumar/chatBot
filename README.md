# ChatBot Mantra - AI Chatbot from Scratch

This repository contains the complete source code for a custom, AI-powered chatbot built for "Fix Mantra," a fictional gadget and appliance repair service. The chatbot is designed to understand user queries, handle multi-step service bookings, remember conversational context, and provide instant support.

This project was built entirely from scratch without relying on third-party NLP services like Google Dialogflow or Microsoft LUIS, offering full control over the AI model and its behavior.

## ✨ Features

*   **Natural Language Understanding (NLU):** The bot uses a custom-trained neural network to understand the *intent* behind a user's message.
*   **Conversational Context Management:** The bot can handle multi-step conversations, such as a complete service booking form, by remembering previous user inputs within a session.
*   **Task-Oriented Dialogue:** Guides users through specific tasks like gathering their name, phone, address, and device problem.
*   **Decoupled Architecture:** A separate React frontend and Python backend allow for independent development and scaling.
*   **Deployment Ready:** The application is configured for a professional, free-tier deployment on cloud platforms like Render and Vercel.

## 💻 Technology Stack

This project utilizes a modern stack for both the AI model and the web application.

*   ### **Backend & API**
    *   **Python:** The core language for the server and AI model.
    *   **Flask:** A lightweight web framework used to create the `/predict` API endpoint.
    *   **Waitress:** A production-ready WSGI server used to run the Flask application, compatible with both Windows and Linux.

*   ### **AI / Machine Learning**
    *   **TensorFlow (Keras):** Used to design, build, and train the deep learning model (a Sequential Neural Network) that classifies user intents.
    *   **NLTK (Natural Language Toolkit):** Used for essential text pre-processing steps like **tokenization** (splitting sentences into words) and **lemmatization** (reducing words to their root form).
    *   **NumPy & Pickle:** Used for efficient numerical data manipulation and for saving the processed vocabulary and classes.

*   ### **Frontend**
    *   **React.js:** A popular JavaScript library for building the dynamic and responsive chat user interface.

## 🏗️ System Architecture

The system is designed with a clear separation of concerns, making it scalable and maintainable.

+----------------+ (1) User message +-------------------------+ (3) Uses +---------------------+
| | (HTTP POST to | | | |
| React Frontend| /predict endpoint) | Python Backend Server | ------------------> | Trained AI Model |
| (Chatbot UI) | --------------------------> | (Flask/Waitress) | | (chatbot_model.h5) |
| | <-------------------------- | | <------------------ | |
+----------------+ (2) AI-generated reply +-------------------------+ (4) Gets Result +---------------------+


---

## 🚀 Getting Started: Running the Project Locally

Follow these steps to set up and run the entire chatbot application on your local machine.

### Prerequisites

Make sure you have the following installed on your system:
*   **Python** (version 3.9, 3.10, or 3.11 is recommended)
*   **Node.js** and **npm** (for the React frontend)
*   **Git** for cloning the repository.

### Step 1: Clone the Repository

First, clone the project to your local machine using Git.
git clone https://github.com/your-username/fixmantra-chatbot.git
cd fixmantra-chatbot

2. Create a Python Virtual Environment:
    python -m venv .venv
3. Activate the Virtual Environment:  .venv\Scripts\activate
   . On Windows (PowerShell):  .venv\Scripts\Activate.ps1
   . On macOS and Linux:  source .venv/bin/activate
4. Install Required Python Libraries:  pip install -r requirements.txt
5. Train the AI Model:  python train.py
6. Run the Backend Server: python app.py
   ##Running Server Port: 8080
 ##view server 
<img width="1631" height="941" alt="image" src="https://github.com/user-attachments/assets/619272c4-65d3-481a-b074-a3f13e7756ae" />

