# Step 3: Import deep learning libraries
print('Step 3: Importing TensorFlow/Keras...')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

print(' TensorFlow/Keras libraries imported')
print(f'TensorFlow version: {tf.__version__}')
