# Step 8: Build LSTM model
print('Step 8: Building LSTM model...')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# Load and prepare data
df = pd.read_csv('../data/raw/Twitter_Data.csv')
df = df.dropna(subset=['clean_text', 'category'])
df['clean_text'] = df['clean_text'].astype(str)
df['sentiment'] = df['category'].map({-1: 0, 0: 1, 1: 2})

X = df['clean_text'].values
y = df['sentiment'].values

# Tokenization
vocab_size = 10000
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X)
X_sequences = tokenizer.texts_to_sequences(X)
X_padded = pad_sequences(X_sequences, maxlen=50, padding='post', truncating='post')

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_padded, y, test_size=0.2, random_state=42, stratify=y
)

# Build LSTM model
embedding_dim = 100  # We'll use random embeddings first

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=50))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))  # 3 classes: negative, neutral, positive

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print(' Model architecture:')
model.summary()

print('\nModel compiled successfully!')
print('Ready for training...')
