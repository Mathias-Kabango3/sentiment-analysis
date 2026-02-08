# Step 10: Train LSTM with Word2Vec Skip-gram embeddings
print('Step 10: Training LSTM with Word2Vec Skip-gram embeddings...')

import pandas as pd
import numpy as np
import time
import gensim
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

print('1. Loading and preparing data...')
df = pd.read_csv('../data/raw/Twitter_Data.csv')
df = df.dropna(subset=['clean_text', 'category'])
df['clean_text'] = df['clean_text'].astype(str)
df['sentiment'] = df['category'].map({-1: 0, 0: 1, 1: 2})

# Prepare text for Word2Vec
sentences = [text.split() for text in df['clean_text'].values]

print('2. Training Word2Vec Skip-gram model...')
start_time = time.time()

# Train Word2Vec Skip-gram
w2v_model = Word2Vec(
    sentences=sentences,
    vector_size=100,      # Embedding dimension
    window=5,            # Context window size
    min_count=2,         # Ignore rare words
    workers=4,           # Number of CPU cores
    sg=1,                # 1 for Skip-gram, 0 for CBOW
    epochs=10
)

w2v_train_time = time.time() - start_time
print(f'Word2Vec training completed in {w2v_train_time:.2f} seconds')
print(f'Vocabulary size in Word2Vec: {len(w2v_model.wv)}')

print('3. Creating embedding matrix...')
# Tokenize text for Keras
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df['clean_text'].values)
vocab_size = min(10000, len(tokenizer.word_index) + 1)

# Create embedding matrix
embedding_dim = 100
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in tokenizer.word_index.items():
    if i >= vocab_size:
        continue
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]
    else:
        # Random initialization for unknown words
        embedding_matrix[i] = np.random.normal(size=(embedding_dim,))

print(f'Embedding matrix shape: {embedding_matrix.shape}')
print(f'Percentage of words found in Word2Vec: {np.sum(np.any(embedding_matrix, axis=1)) / vocab_size * 100:.2f}%')

print('4. Preparing sequences...')
X_sequences = tokenizer.texts_to_sequences(df['clean_text'].values)
X_padded = pad_sequences(X_sequences, maxlen=50, padding='post', truncating='post')
y = df['sentiment'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_padded, y, test_size=0.2, random_state=42, stratify=y
)

print('5. Building LSTM with Word2Vec embeddings...')
model = Sequential()
model.add(Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dim,
    weights=[embedding_matrix],
    input_length=50,
    trainable=False  # Keep Word2Vec embeddings fixed
))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print('6. Training LSTM...')
start_time = time.time()

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=64,
    verbose=1
)

lstm_train_time = time.time() - start_time
print(f'LSTM training completed in {lstm_train_time:.2f} seconds')

print('7. Evaluation...')
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Accuracy: {test_acc:.4f}')
print(f'Test Loss: {test_loss:.4f}')

y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

print('\n=== CLASSIFICATION REPORT ===')
print(classification_report(y_test, y_pred_classes, target_names=['Negative', 'Neutral', 'Positive']))

print(' LSTM with Word2Vec Skip-gram embeddings complete!')
print(f'Total time: {w2v_train_time + lstm_train_time:.2f} seconds')
