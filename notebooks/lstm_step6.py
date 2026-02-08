# Step 6: Text tokenization
print('Step 6: Tokenizing text...')

import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load and prepare data
df = pd.read_csv('../data/raw/Twitter_Data.csv')
df = df.dropna(subset=['clean_text', 'category'])
df['clean_text'] = df['clean_text'].astype(str)
df['sentiment'] = df['category'].map({-1: 0, 0: 1, 1: 2})

X = df['clean_text'].values
y = df['sentiment'].values

# Tokenization
tokenizer = Tokenizer(num_words=10000)  # Keep top 10,000 words
tokenizer.fit_on_texts(X)

# Convert text to sequences
X_sequences = tokenizer.texts_to_sequences(X)

# Check vocabulary
word_index = tokenizer.word_index
print(f'Vocabulary size: {len(word_index)}')
print(f'Number of sequences: {len(X_sequences)}')

# Check sequence lengths
seq_lengths = [len(seq) for seq in X_sequences]
print(f'Average sequence length: {np.mean(seq_lengths):.1f}')
print(f'Maximum sequence length: {max(seq_lengths)}')
print(f'Minimum sequence length: {min(seq_lengths)}')

# Pad sequences (we'll decide maxlen later)
max_length = 50  # Based on average ~20 words
X_padded = pad_sequences(X_sequences, maxlen=max_length, padding='post', truncating='post')

print(f'X_padded shape: {X_padded.shape}')
print(' Tokenization complete!')
