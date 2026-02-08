# Twitter Sentiment Analysis

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A comprehensive comparative study of Machine Learning and Deep Learning approaches for Twitter sentiment classification. This project evaluates traditional ML models against neural network architectures with various word embedding techniques.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Models Implemented](#models-implemented)
- [Results Summary](#results-summary)
- [Installation](#installation)
- [Usage](#usage)
- [Notebooks](#notebooks)
- [Team](#team)
- [License](#license)

##  Overview

This project implements and compares multiple approaches for sentiment analysis on Twitter data:

- **Traditional ML Models**: Naive Bayes, Logistic Regression
- **Deep Learning Models**: Vanilla RNN, LSTM, GRU
- **Word Embeddings**: Trainable, Word2Vec (Skip-gram & CBOW), GloVe, FastText

The goal is to determine the most effective combination of model architecture and text representation for classifying tweets into three sentiment categories: Negative, Neutral, and Positive.

##  Project Structure

```
sentiment-analysis/
│
├── data/
│   ├── raw/                    # Original Twitter dataset
│   │   └── Twitter_Data.csv
│   └── processed/              # Cleaned and preprocessed data
│
├── notebooks/
│   ├── 01.eda.ipynb                      # Exploratory Data Analysis
│   ├── 02.text_cleaning.ipynb            # Text Preprocessing Pipeline
│   ├── 03.model_baseline.ipynb           # Baseline ML Models (NB, LR)
│   ├── 04.model_experiments.ipynb        # Vanilla RNN Experiments
│   └── 05.GRU_model_and_experiments.ipynb # GRU Model Experiments
│
├── src/
│   ├── data/
│   │   ├── clean_text.py       # Text cleaning functions
│   │   ├── load_data.py        # Data loading utilities
│   │   └── save_data.py        # Data saving utilities
│   ├── features/
│   │   └── build_features.py   # Feature engineering
│   ├── models/                 # Model implementations
│   ├── pipelines/              # Training pipelines
│   └── utils/                  # Utility functions
│
├── models/                     # Saved trained models
│
├── reports/                    # Generated reports and visualizations
│   ├── Sentiment_Analysis_Research_Report.docx
│   └── embedding_comparison.png
│
├── tests/                      # Unit tests
│
├── requirements.txt            # Project dependencies
└── README.md                   # This file
```

## Dataset

The project uses the **Twitter Sentiment Analysis Dataset** containing tweets labeled with sentiment polarity.

| Attribute | Value |
|-----------|-------|
| Total Samples | 162,969 |
| Negative Tweets | 35,509 (21.8%) |
| Neutral Tweets | 55,213 (33.9%) |
| Positive Tweets | 72,247 (44.3%) |
| Text Column | `clean_text` |

### Label Encoding
- `-1` → Negative
- `0` → Neutral  
- `1` → Positive

## 🤖 Models Implemented

### Baseline Models (Traditional ML)

| Model | Feature Extraction |
|-------|-------------------|
| Naive Bayes | Count Vectorizer, TF-IDF |
| Logistic Regression | Count Vectorizer, TF-IDF |

### Deep Learning Models

| Model | Architecture | Embeddings |
|-------|-------------|------------|
| Vanilla RNN | 2-layer SimpleRNN (128→64 units) | Trainable, Word2Vec, GloVe, FastText |
| GRU | Bidirectional GRU (64 units) | Word2Vec Skip-gram, CBOW, TF-IDF |
| LSTM | Bidirectional LSTM | Word2Vec Skip-gram, CBOW |

### Word Embedding Configurations

| Embedding | Dimension | Description |
|-----------|-----------|-------------|
| Trainable | 100 | Learned during training |
| Word2Vec Skip-gram | 100/128 | Predicts context from target word |
| Word2Vec CBOW | 100/128 | Predicts target from context words |
| GloVe Twitter | 100 | Pre-trained on 2B tweets |
| FastText | 100 | Character n-gram embeddings |

## Results Summary

### Best Performing Models

| Rank | Model | Embedding | Accuracy | F1-Score |
|------|-------|-----------|----------|----------|
| 1 | **GRU** | Skip-gram | **63.93%** | **0.6316** |
| 2 | GRU | CBOW | 63.12% | 0.6267 |
| 3 | Logistic Regression | TF-IDF | 60.89% | 0.6021 |
| 4 | Vanilla RNN | Word2Vec | 59.94% | 0.5891 |

### Key Findings

1. **Deep Learning Superiority**: GRU models outperformed traditional ML approaches by ~3% accuracy
2. **Best Embedding**: Word2Vec Skip-gram consistently provided the best results
3. **Bidirectional Advantage**: Bidirectional architectures captured context more effectively
4. **Competitive Baseline**: Logistic Regression with TF-IDF remains a strong baseline

## Installation

### Prerequisites

- Python 3.10+
- pip or conda

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Mathias-Kabango3/sentiment-analysis.git
   cd sentiment-analysis
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK resources**
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('punkt')
   ```

## Usage

### Running Notebooks

Launch Jupyter and run notebooks in order:

```bash
jupyter notebook
```

1. `01.eda.ipynb` - Explore the dataset
2. `02.text_cleaning.ipynb` - Preprocess text data
3. `03.model_baseline.ipynb` - Train baseline models
4. `04.model_experiments.ipynb` - Train Vanilla RNN
5. `05.GRU_model_and_experiments.ipynb` - Train GRU models

### Quick Start - Training a Model

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout, Bidirectional

# Build GRU model
model = Sequential([
    Embedding(vocab_size, embedding_dim, weights=[embedding_matrix]),
    Bidirectional(GRU(64, dropout=0.2)),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1)
```

### Loading Saved Models

```python
from tensorflow.keras.models import load_model
import pickle

# Load model
model = load_model('models/rnn_best_model.keras')

# Load tokenizer
with open('models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Predict
def predict_sentiment(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=100, padding='post')
    prediction = model.predict(padded)
    labels = ['Negative', 'Neutral', 'Positive']
    return labels[prediction.argmax()]

print(predict_sentiment("I love this product!"))  # Output: Positive
```

## 📓 Notebooks

| Notebook | Description |
|----------|-------------|
| `01.eda.ipynb` | Exploratory data analysis: distribution plots, word clouds, text statistics |
| `02.text_cleaning.ipynb` | Text preprocessing: cleaning, tokenization, lemmatization |
| `03.model_baseline.ipynb` | Baseline models: Naive Bayes & Logistic Regression with BoW/TF-IDF |
| `04.model_experiments.ipynb` | Vanilla RNN with Trainable, Word2Vec, GloVe, FastText embeddings |
| `05.GRU_model_and_experiments.ipynb` | GRU models with Skip-gram, CBOW, TF-IDF features |

## 📦 Dependencies

```
tensorflow>=2.10.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
gensim>=4.0.0
nltk>=3.6.0
matplotlib>=3.4.0
seaborn>=0.11.0
wordcloud>=1.8.0
python-docx>=0.8.11
```


## 📄 Reports

The complete research report is available at:
- `reports/Sentiment_Analysis_Research_Report.docx`

## 🔮 Future Work

- [ ] Implement Transformer-based models (BERT, RoBERTa)
- [ ] Add attention mechanisms to RNN models
- [ ] Develop real-time sentiment analysis API
- [ ] Extend to multilingual sentiment analysis
- [ ] Implement aspect-based sentiment analysis



## 🙏 Acknowledgments

- Stanford NLP Group for GloVe embeddings
- Facebook AI Research for FastText
- The TensorFlow and Keras teams
- Twitter for providing the dataset

---

<p align="center">
  Made with ❤️ 
</p>
