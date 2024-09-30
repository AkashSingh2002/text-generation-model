import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model
model = load_model('text_generation_model_gutenberg.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to generate text
def generate_text(model, tokenizer, input_text, max_sequence_len, num_words_to_generate=20):
    for _ in range(num_words_to_generate):
        token_list = tokenizer.texts_to_sequences([input_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=1)[0]
        predicted_word = tokenizer.index_word.get(predicted_word_index, '')
        input_text += ' ' + predicted_word
    return input_text

# Provide a prompt for the model to generate text
input_prompt = "Alice replied thoughtfully, 'I wonder if..."
max_sequence_len = 50  # Adjust according to the maximum sequence length used during training
generated_text = generate_text(model, tokenizer, input_prompt, max_sequence_len)
print(generated_text)
