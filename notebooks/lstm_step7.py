# Step 7: Train-test split
print('Step 7: Splitting data...')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X)
X_sequences = tokenizer.texts_to_sequences(X)
X_padded = pad_sequences(X_sequences, maxlen=50, padding='post', truncating='post')

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_padded, y, test_size=0.2, random_state=42, stratify=y
)

print(f'Training samples: {X_train.shape[0]}')
print(f'Testing samples: {X_test.shape[0]}')
print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')

print(' Data split complete!')
