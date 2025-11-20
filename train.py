import json
import pickle
import random

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
# Some NLTK installations expect additional punkt variants; ensure they're present
try:
    nltk.download('punkt_tab')
except Exception:
    # If the downloader doesn't know 'punkt_tab', ignore and continue
    pass

# Optionally download the Open Multilingual Wordnet data used by some lemmatizers
try:
    nltk.download('omw-1.4')
except Exception:
    pass

lemmatizer = WordNetLemmatizer()

# Load the intents file
with open('intents.json') as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Process each intent and its patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize words - split sentence into words. If the punkt/punkt_tab
        # tokenizer is missing, attempt to download it and retry once.
        try:
            word_list = nltk.word_tokenize(pattern)
        except LookupError:
            nltk.download('punkt')
            try:
                nltk.download('punkt_tab')
            except Exception:
                pass
            word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        # Add the tag to the classes list if not already there
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize words, convert to lowercase, and remove ignored letters
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Save the processed words and classes to files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# --- Convert text data into numerical data (Bag of Words) ---
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

# Split into training input (X) and training output (Y)
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# --- Build the Neural Network ---
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save the trained model
model.save('chatbot_model.h5')
print("Done! Model is trained and saved.")