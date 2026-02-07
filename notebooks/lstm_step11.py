# Step 11: Train LSTM with Word2Vec CBOW embeddings
print('Step 11: Training LSTM with Word2Vec CBOW embeddings...')

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

print('1. Loading data...')
df = pd.read_csv('../data/raw/Twitter_Data.csv')
df = df.dropna(subset=['clean_text', 'category'])
df['clean_text'] = df['clean_text'].astype(str)
df['sentiment'] = df['category'].map({-1: 0, 0: 1, 1: 2})

# Prepare text
sentences = [text.split() for text in df['clean_text'].values]

print('2. Training Word2Vec CBOW model...')
start_time = time.time()

# Train Word2Vec CBOW (sg=0 for CBOW)
w2v_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    sg=0,  # 0 for CBOW, 1 for Skip-gram
    epochs=10
)

w2v_train_time = time.time() - start_time
print(f'Word2Vec CBOW training: {w2v_train_time:.2f} seconds')
print(f'Vocabulary size: {len(w2v_model.wv)}')

print('3. Creating embedding matrix...')
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df['clean_text'].values)
vocab_size = min(10000, len(tokenizer.word_index) + 1)

embedding_dim = 100
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in tokenizer.word_index.items():
    if i >= vocab_size:
        continue
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]
    else:
        embedding_matrix[i] = np.random.normal(size=(embedding_dim,))

coverage = np.sum(np.any(embedding_matrix, axis=1)) / vocab_size * 100
print(f'Embedding matrix: {embedding_matrix.shape}')
print(f'Word coverage: {coverage:.2f}%')

print('4. Preparing sequences...')
X_sequences = tokenizer.texts_to_sequences(df['clean_text'].values)
X_padded = pad_sequences(X_sequences, maxlen=50, padding='post', truncating='post')
y = df['sentiment'].values

X_train, X_test, y_train, y_test = train_test_split(
    X_padded, y, test_size=0.2, random_state=42, stratify=y
)

print('5. Building LSTM with CBOW embeddings...')
model = Sequential()
model.add(Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dim,
    weights=[embedding_matrix],
    input_length=50,
    trainable=False  # Fixed embeddings
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

print('6. Training...')
start_time = time.time()

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=64,
    verbose=1
)

lstm_train_time = time.time() - start_time
print(f'LSTM training: {lstm_train_time:.2f} seconds')

print('7. Evaluation...')
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Accuracy: {test_acc:.4f}')
print(f'Test Loss: {test_loss:.4f}')

y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

print('\n=== CLASSIFICATION REPORT ===')
print(classification_report(y_test, y_pred_classes, target_names=['Negative', 'Neutral', 'Positive']))

print(' LSTM with Word2Vec CBOW embeddings complete!')
print(f'Total time: {w2v_train_time + lstm_train_time:.2f} seconds')

# Save results for comparison
results = {
    'method': 'Word2Vec CBOW',
    'accuracy': test_acc,
    'loss': test_loss,
    'total_time': w2v_train_time + lstm_train_time
}
print(f'\nResults saved: {results}')
