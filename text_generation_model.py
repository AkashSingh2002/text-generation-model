import requests
import re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense
import pickle

# Step 1: Download text from Project Gutenberg
url = "https://www.gutenberg.org/files/11/11-0.txt"  # Alice's Adventures in Wonderland
response = requests.get(url)
text = response.text

# Step 2: Clean the downloaded text
def preprocess_text(text):
    # Remove the Project Gutenberg header/footer
    start_index = text.find("CHAPTER I")
    end_index = text.rfind("THE END")
    text = text[start_index:end_index]
    
    # Lowercase and remove non-alphabetical characters
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Keep only alphabets and spaces
    return text

cleaned_text = preprocess_text(text)

# Step 3: Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([cleaned_text])
total_words = len(tokenizer.word_index) + 1

# Create input sequences for the model
input_sequences = []
for line in cleaned_text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad the sequences for a consistent input length
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

# Split into input (X) and output (y)
X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = to_categorical(y, num_classes=total_words)

# Step 4: Build the model
model = Sequential()
model.add(Embedding(input_dim=total_words, output_dim=100, input_length=max_sequence_len-1))
model.add(LSTM(150, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=100, batch_size=64, verbose=1)

# Save the trained model
model.save('text_generation_model_gutenberg.h5')

# Step 5: Save the tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
