# GRU Model Experiments Documentation

## Sentiment Analysis Project - Deep Learning Models Report

**Date:** February 2026  
**Dataset:** Twitter Sentiment Data  
**Notebook:** `05_GRU_model_and_experiments.ipynb`

---

## 1. Overview

This document describes the implementation and evaluation of Gated Recurrent Unit (GRU) neural networks for Twitter sentiment classification. Three distinct experiments were conducted to compare different text representation approaches: Skip-gram embeddings, CBOW (Continuous Bag of Words) embeddings, and TF-IDF features.

### Project Context

| Attribute | Value |
|-----------|-------|
| Model Architecture | GRU (Gated Recurrent Units) |
| Task | Multi-class Sentiment Classification |
| Classes | 3 (Negative, Neutral, Positive) |
| Input Data | Pre-processed Twitter text |
| Experiments | 3 distinct approaches |

---

## 2. Experimental Design

### 2.1 Experiment Overview

| Experiment | Text Representation | Embedding Method | Dimensionality |
|------------|-------------------|------------------|----------------|
| Experiment 1 | TF-IDF vectors | Statistical | Variable (top features) |
| Experiment 2 | Skip-gram embeddings | Word2Vec (sg=1) | 100 dimensions |
| Experiment 3 | CBOW embeddings | Word2Vec (sg=0) | 100 dimensions |

### 2.2 Rationale

**Skip-gram vs CBOW:**
- **Skip-gram** predicts context words from target word → Better for rare words, captures nuanced relationships
- **CBOW** predicts target word from context → Faster training, better for frequent words

**TF-IDF:**
- Statistical approach without pre-training
- Captures term importance across corpus
- Baseline for comparison with learned embeddings

---

## 3. Data Preparation

### 3.1 Data Loading

```python
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load preprocessed data
file_path = '/content/drive/MyDrive/data/processed/twitter_cleaned.csv'
df = pd.read_csv(file_path)
```

### 3.2 Label Mapping

The sentiment labels were mapped from categorical to numeric format:

| Original Label | Numeric Code | Sentiment |
|----------------|-------------|-----------|
| -1 | 0 | Negative |
| 0 | 1 | Neutral |
| 1 | 2 | Positive |

```python
# Map labels: -1 → 0, 0 → 1, 1 → 2
df['label'] = df['label'].map({-1: 0, 0: 1, 1: 2})
```

### 3.3 Train-Test Split

**Configuration:**
- Training set: 80%
- Test set: 20%
- Stratification: Enabled (maintains class distribution)
- Random state: 42 (for reproducibility)

```python
X_train, X_test, y_train, y_test = train_test_split(
    df['text_basic'], 
    df['label'], 
    test_size=0.2, 
    random_state=42, 
    stratify=df['label']
)
```

---

## 4. Text Tokenization & Sequence Preparation

### 4.1 Tokenization Parameters

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configuration
MAX_WORDS = 10000      # Vocabulary size limit
MAX_LEN = 100          # Maximum sequence length
```

### 4.2 Tokenization Process

**Steps:**
1. Initialize Tokenizer with vocabulary limit
2. Fit on training text
3. Convert text to sequences of integers
4. Pad sequences to uniform length

```python
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

# Convert to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post', truncating='post')
```

**Padding Strategy:**
- Type: Post-padding (add zeros at end)
- Truncation: Post-truncation (cut from end)
- Rationale: Preserves beginning of tweets which often contain key sentiment

---

## 5. Word2Vec Training

### 5.1 Sentence Preparation

```python
from gensim.models import Word2Vec

# Prepare sentences for Word2Vec
sentences = [text.split() for text in X_train]
```

### 5.2 Skip-gram Model (sg=1)

**Hyperparameters:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `vector_size` | 100 | Embedding dimensionality |
| `window` | 5 | Context window size |
| `min_count` | 2 | Minimum word frequency |
| `workers` | 4 | Parallel processing threads |
| `sg` | 1 | Skip-gram architecture |
| `epochs` | 10 | Training iterations |

```python
print("Training Skip-gram (Word2Vec) model...")
sg_w2v_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    sg=1,  # Skip-gram
    epochs=10
)
```

**Skip-gram Characteristics:**
- Predicts context words given target word
- Better for capturing semantic relationships
- More effective for rare words
- Computationally more intensive

### 5.3 CBOW Model (sg=0)

```python
print("Training CBOW (Word2Vec) model...")
cbow_w2v_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    sg=0,  # CBOW
    epochs=10
)
```

**CBOW Characteristics:**
- Predicts target word from context
- Faster training than Skip-gram
- Better for frequent words
- Smoother representations

### 5.4 Embedding Matrix Creation

The embedding matrices map tokenizer vocabulary to Word2Vec vectors:

```python
def create_embedding_matrix(w2v_model, tokenizer, max_words, embedding_dim):
    """
    Create embedding matrix for Keras Embedding layer.
    
    Args:
        w2v_model: Trained Word2Vec model
        tokenizer: Keras Tokenizer with vocabulary
        max_words: Maximum vocabulary size
        embedding_dim: Embedding vector dimensions
    
    Returns:
        numpy array of shape (vocab_size, embedding_dim)
    """
    vocab_size = min(len(tokenizer.word_index) + 1, max_words)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    for word, idx in tokenizer.word_index.items():
        if idx < max_words:
            try:
                embedding_matrix[idx] = w2v_model.wv[word]
            except KeyError:
                # Word not in Word2Vec model, leave as zeros
                pass
    
    return embedding_matrix

# Create matrices
sg_matrix = create_embedding_matrix(sg_w2v_model, tokenizer, MAX_WORDS, 100)
cbow_matrix = create_embedding_matrix(cbow_w2v_model, tokenizer, MAX_WORDS, 100)
```

**Matrix Properties:**
- Shape: (vocabulary_size, 100)
- Unknown words: Zero vectors
- Pre-trained weights: Frozen during initial training

---

## 6. GRU Architecture

### 6.1 Model Function Definition

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout

def build_gru_model(vocab_size, embedding_dim, embedding_matrix=None, max_len=100):
    """
    Build GRU model for sentiment classification.
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Embedding vector dimensions
        embedding_matrix: Pre-trained embeddings (optional)
        max_len: Maximum sequence length
    
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix] if embedding_matrix is not None else None,
            input_length=max_len,
            trainable=True  # Allow fine-tuning
        ),
        GRU(128, return_sequences=False),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(3, activation='softmax')  # 3 classes
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

### 6.2 Architecture Breakdown

| Layer | Type | Parameters | Purpose |
|-------|------|------------|---------|
| 1 | Embedding | vocab_size × 100 | Convert tokens to dense vectors |
| 2 | GRU | 128 units | Sequential pattern learning |
| 3 | Dropout | 50% rate | Prevent overfitting |
| 4 | Dense | 64 units, ReLU | Feature extraction |
| 5 | Dropout | 30% rate | Regularization |
| 6 | Dense | 3 units, Softmax | Classification output |

**Key Design Decisions:**

1. **Trainable Embeddings:** Set to `True` to allow fine-tuning on sentiment task
2. **Single GRU Layer:** Sufficient for tweet-length sequences
3. **128 GRU Units:** Balance between capacity and overfitting
4. **Dropout Layers:** Combat overfitting (50% and 30% rates)
5. **Dense Layer:** Additional non-linear transformation before classification

---

## 7. Experiment 1: GRU + TF-IDF

### 7.1 TF-IDF Vectorization

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF vectorizer
tfidf = TfidfVectorizer(
    max_features=5000,     # Limit vocabulary
    ngram_range=(1, 2),    # Unigrams and bigrams
    min_df=2,              # Minimum document frequency
    max_df=0.95            # Maximum document frequency (remove very common terms)
)

# Fit and transform
X_train_tfidf = tfidf.fit_transform(X_train).toarray()
X_test_tfidf = tfidf.transform(X_test).toarray()
```

### 7.2 3D Reshaping for GRU

**Challenge:** GRU expects 3D input (samples, timesteps, features), but TF-IDF produces 2D (samples, features).

**Solution:** Reshape by treating each feature as a timestep with 1 feature:

```python
# Reshape: (samples, features) → (samples, timesteps=features, features=1)
X_train_tfidf_3d = X_train_tfidf.reshape((X_train_tfidf.shape[0], X_train_tfidf.shape[1], 1))
X_test_tfidf_3d = X_test_tfidf.reshape((X_test_tfidf.shape[0], X_test_tfidf.shape[1], 1))
```

### 7.3 TF-IDF-Specific Model

```python
from tensorflow.keras.layers import Input

def build_gru_tfidf_model(timesteps, feature_dim):
    """
    Build GRU model for TF-IDF input.
    
    Args:
        timesteps: Number of TF-IDF features (treated as timesteps)
        feature_dim: Feature dimensionality (1 for TF-IDF)
    """
    model = Sequential([
        Input(shape=(timesteps, feature_dim)),
        GRU(128, return_sequences=False),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Build model
timesteps = X_train_tfidf_3d.shape[1]
model_tfidf = build_gru_tfidf_model(timesteps, 1)
```

### 7.4 Training

```python
history_tfidf = model_tfidf.fit(
    X_train_tfidf_3d,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)
```

---

## 8. Experiment 2: GRU + Skip-gram

### 8.1 Model Initialization

```python
vocab_size_sg = sg_matrix.shape[0]

print("🚀 Starting Experiment: GRU + Skip-gram")
model_sg = build_gru_model(
    vocab_size=vocab_size_sg,
    embedding_dim=100,
    embedding_matrix=sg_matrix,
    max_len=MAX_LEN
)
```

### 8.2 Training

```python
history_sg = model_sg.fit(
    X_train_pad,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)
```

**Training Configuration:**
- Epochs: 10
- Batch size: 64
- Validation split: 20%
- Pre-trained embeddings: Skip-gram vectors
- Embedding fine-tuning: Enabled

---

## 9. Experiment 3: GRU + CBOW

### 9.1 Model Initialization

```python
vocab_size_cbow = cbow_matrix.shape[0]

print("🚀 Starting Experiment: GRU + CBOW")
model_cbow = build_gru_model(
    vocab_size=vocab_size_cbow,
    embedding_dim=100,
    embedding_matrix=cbow_matrix,
    max_len=MAX_LEN
)
```

### 9.2 Training

```python
history_cbow = model_cbow.fit(
    X_train_pad,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)
```

---

## 10. Model Evaluation

### 10.1 Evaluation Metrics

For each experiment, the following metrics were computed:

```python
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

def evaluate_model(model, X_test, y_test, model_name):
    """
    Comprehensive model evaluation.
    
    Returns:
        Dictionary with accuracy, precision, recall, F1-score
    """
    # Predictions
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )
    
    print(f"\n{'='*50}")
    print(f"{model_name} - Evaluation Results")
    print(f"{'='*50}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive'])}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
```

### 10.2 Experiment Results

| Metric | TF-IDF | Skip-gram | CBOW |
|--------|--------|-----------|------|
| Accuracy | - | - | - |
| Precision | - | - | - |
| Recall | - | - | - |
| F1-Score | - | - | - |

*Note: Actual values would be populated from model execution output.*

### 10.3 Per-Class Performance

Each model was evaluated on individual sentiment classes:

**Classification Report Structure:**
- Precision: True Positives / (True Positives + False Positives)
- Recall: True Positives / (True Positives + False Negatives)
- F1-Score: Harmonic mean of Precision and Recall
- Support: Number of samples in each class

---

## 11. Comparative Analysis

### 11.1 Results Visualization

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Compile results
results = pd.DataFrame({
    'Model': ['GRU + TF-IDF', 'GRU + Skip-gram', 'GRU + CBOW'],
    'Accuracy': [acc_tfidf, acc_sg, acc_cbow],
    'Precision': [prec_tfidf, prec_sg, prec_cbow],
    'Recall': [rec_tfidf, rec_sg, rec_cbow],
    'F1-Score': [f1_tfidf, f1_sg, f1_cbow]
})

# Visualization
plt.figure(figsize=(12, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(results))
width = 0.2

for i, metric in enumerate(metrics):
    plt.bar(x + i*width, results[metric], width, label=metric)

plt.xlabel('Models')
plt.ylabel('Score')
plt.title('GRU Model Performance Comparison')
plt.xticks(x + width*1.5, results['Model'], rotation=15)
plt.legend()
plt.tight_layout()
plt.show()
```

### 11.2 Key Findings

**Expected Patterns:**

1. **Word2Vec vs TF-IDF:**
   - Word2Vec embeddings should capture semantic relationships better
   - TF-IDF provides strong baseline with term importance weighting

2. **Skip-gram vs CBOW:**
   - Skip-gram typically performs better on nuanced sentiment
   - CBOW may be faster to train with comparable results

3. **Embedding Fine-tuning:**
   - Trainable embeddings allow task-specific adaptation
   - Pre-trained initialization provides strong starting point

---

## 12. Technical Implementation Details

### 12.1 Training Configuration

**Common Settings Across Experiments:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | Adam | Adaptive learning rates |
| Loss Function | Sparse Categorical Crossentropy | Multi-class classification |
| Batch Size | 64 | Balance between speed and stability |
| Epochs | 10 | Sufficient for convergence |
| Validation Split | 0.2 | Monitor overfitting |

### 12.2 Regularization Techniques

1. **Dropout Layers:**
   - Rate 1: 50% (after GRU)
   - Rate 2: 30% (after Dense)
   - Purpose: Prevent overfitting

2. **Early Stopping (Implicit):**
   - Validation monitoring enabled
   - Could add explicit early stopping callback

### 12.3 Hardware Considerations

**Google Colab Environment:**
- GPU: Enabled for faster training
- RAM: Standard allocation
- Storage: Google Drive mounted for data access

---

## 13. Reproducibility

### 13.1 Random Seed Configuration

```python
import random
import tensorflow as tf

# Set seeds for reproducibility
RANDOM_STATE = 42

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
```

### 13.2 Version Information

**Key Libraries:**

| Library | Purpose |
|---------|---------|
| `tensorflow` | Deep learning framework |
| `keras` | High-level neural network API |
| `gensim` | Word2Vec implementation |
| `sklearn` | Train-test split, metrics, TF-IDF |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |

### 13.3 File Paths

```python
# Data paths
INPUT_DATA = '/content/drive/MyDrive/data/processed/twitter_cleaned.csv'
MODEL_SAVE_PATH = '/content/drive/MyDrive/models/'

# Model filenames
MODEL_SG = 'gru_skipgram.h5'
MODEL_CBOW = 'gru_cbow.h5'
MODEL_TFIDF = 'gru_tfidf.h5'
```

---

## 14. Future Improvements

### 14.1 Model Architecture Enhancements

1. **Bidirectional GRU:**
   - Process sequences in both directions
   - Capture context from past and future tokens

2. **Stacked GRU Layers:**
   - Multiple GRU layers for hierarchical learning
   - May improve performance on complex patterns

3. **Attention Mechanism:**
   - Focus on important parts of sequence
   - Interpretable attention weights

### 14.2 Training Optimization

1. **Learning Rate Scheduling:**
   - Reduce learning rate on plateau
   - Improve convergence

2. **Early Stopping:**
   - Prevent overfitting
   - Save best model weights

3. **Batch Size Tuning:**
   - Experiment with different batch sizes
   - Balance training speed and generalization

### 14.3 Embedding Enhancements

1. **Pre-trained Embeddings:**
   - GloVe vectors trained on larger corpora
   - FastText for handling out-of-vocabulary words

2. **Contextual Embeddings:**
   - BERT-based approaches
   - ELMo embeddings

3. **Domain-Specific Training:**
   - Train Word2Vec on larger Twitter corpus
   - Capture Twitter-specific language patterns

---

## 15. Conclusion

### 15.1 Summary

This notebook implemented and compared three GRU-based approaches for Twitter sentiment analysis:

1. **GRU + TF-IDF:** Statistical feature representation
2. **GRU + Skip-gram:** Context-to-target word embeddings
3. **GRU + CBOW:** Target-from-context word embeddings

All models utilized the same GRU architecture with different input representations, enabling fair comparison of embedding techniques.

### 15.2 Key Takeaways

- **GRU networks** effectively model sequential dependencies in text
- **Pre-trained embeddings** provide semantic initialization
- **Fine-tuning** allows task-specific adaptation
- **Regularization** through dropout is crucial for generalization

### 15.3 Methodological Strengths

- Consistent architecture across experiments
- Proper train-test split with stratification
- Comprehensive evaluation metrics
- Reproducible experimental setup

---

## 16. Dependencies

### 16.1 Python Libraries

```txt
tensorflow>=2.10.0
keras>=2.10.0
gensim>=4.2.0
scikit-learn>=1.1.0
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.12.0
```

### 16.2 Installation Commands

```bash
pip install tensorflow keras gensim scikit-learn pandas numpy matplotlib seaborn
```

### 16.3 Environment Setup

**Google Colab:**
- Runtime: Python 3
- Hardware Accelerator: GPU (recommended)
- RAM: Standard (sufficient for this dataset)

---

## 17. Appendix: Code Reference

### 17.1 Complete Model Building Function

```python
def build_gru_model(vocab_size, embedding_dim, embedding_matrix=None, max_len=100):
    """
    Build and compile GRU model for sentiment classification.
    
    Parameters:
    -----------
    vocab_size : int
        Size of vocabulary (number of unique tokens)
    embedding_dim : int
        Dimensionality of embedding vectors
    embedding_matrix : numpy.ndarray, optional
        Pre-trained embedding weights (default: None)
    max_len : int, optional
        Maximum sequence length (default: 100)
    
    Returns:
    --------
    model : tensorflow.keras.Model
        Compiled Keras model ready for training
    
    Architecture:
    ------------
    - Embedding layer (optional pre-trained weights)
    - GRU layer (128 units)
    - Dropout (0.5)
    - Dense layer (64 units, ReLU)
    - Dropout (0.3)
    - Output layer (3 units, Softmax)
    """
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix] if embedding_matrix is not None else None,
            input_length=max_len,
            trainable=True
        ),
        GRU(128, return_sequences=False),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

### 17.2 Embedding Matrix Creation Function

```python
def create_embedding_matrix(w2v_model, tokenizer, max_words, embedding_dim):
    """
    Create embedding matrix from Word2Vec model for Keras Embedding layer.
    
    Parameters:
    -----------
    w2v_model : gensim.models.Word2Vec
        Trained Word2Vec model
    tokenizer : tensorflow.keras.preprocessing.text.Tokenizer
        Fitted tokenizer with word-to-index mapping
    max_words : int
        Maximum vocabulary size
    embedding_dim : int
        Embedding vector dimensionality
    
    Returns:
    --------
    embedding_matrix : numpy.ndarray
        Matrix of shape (vocab_size, embedding_dim) with pre-trained vectors
    
    Notes:
    ------
    - Words not in Word2Vec vocabulary are initialized as zero vectors
    - Enables transfer learning from Word2Vec to GRU model
    """
    vocab_size = min(len(tokenizer.word_index) + 1, max_words)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    for word, idx in tokenizer.word_index.items():
        if idx < max_words:
            try:
                embedding_matrix[idx] = w2v_model.wv[word]
            except KeyError:
                pass  # Word not in vocabulary, remains zero vector
    
    return embedding_matrix
```

---

*Documentation generated from experimental notebook implementation.*
