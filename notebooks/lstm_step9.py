# Step 9: Train LSTM with random embeddings
print('Step 9: Training LSTM with random embeddings...')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import time

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

# Build LSTM model with random embeddings
embedding_dim = 100

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=50))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print('Starting training...')
start_time = time.time()

# Train with early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

training_time = time.time() - start_time
print(f'Training completed in {training_time:.2f} seconds')

# Evaluate
print('\n=== EVALUATION ===')
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Accuracy: {test_acc:.4f}')
print(f'Test Loss: {test_loss:.4f}')

# Predictions
y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

print('\n=== CLASSIFICATION REPORT ===')
print(classification_report(y_test, y_pred_classes, target_names=['Negative', 'Neutral', 'Positive']))

print(' LSTM with random embeddings complete!')
