import nltk
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

nltk.download('punkt_tab')

# Load dataset
file_path = 'next_word_predictor.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()
 

# Text Cleaning
def separate_punc(doc_text):
    sentences = sent_tokenize(doc_text)  # Split into sentences
    cleaned_sentences = []
    
    for sent in sentences:
        words = word_tokenize(sent)  
        words = [word.lower() for word in words if word.isalnum()] 
        cleaned_sentences.append(" ".join(words))  
    
    return cleaned_sentences  

cleaned_data = separate_punc(text)

# Train-Validation-Test Split
train_texts, temp_texts = train_test_split(cleaned_data, test_size=0.2, random_state=42)
val_texts, test_texts = train_test_split(temp_texts, test_size=0.5, random_state=42)

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(cleaned_data)

# Convert to Sequences
test_sequences = []
for sentence in test_texts:
    tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0]
    for i in range(1, len(tokenized_sentence)):
        test_sequences.append(tokenized_sentence[:i+1])

# Pad Sequences
X_test = pad_sequences(test_sequences, maxlen=66, padding='pre')

# Prepare X_test, y_test
X_test, y_test = X_test[:, :-1], X_test[:, -1]
y_test = to_categorical(y_test, num_classes=4695)

# Save Processed Data
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("Preprocessing complete! Data saved.")
