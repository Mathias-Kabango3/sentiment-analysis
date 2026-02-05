# Text Cleaning Documentation

## Sentiment Analysis Project - Data Preprocessing Report

**Date:** February 2026  
**Dataset:** Twitter Sentiment Data  
**Notebook:** `02.text_cleaning.ipynb`

---

## 1. Overview

This document describes the text cleaning and preprocessing methods applied to the Twitter sentiment analysis dataset. The goal is to prepare the text data for various word embedding techniques and model architectures.

### Dataset Information

| Attribute | Value |
|-----------|-------|
| Source File | `data/raw/Twitter_Data.csv` |
| Total Records | ~192,000 tweets |
| Columns | `clean_text` (tweet content), `category` (sentiment label) |
| Sentiment Labels | -1 (Negative), 0 (Neutral), 1 (Positive) |

---

## 2. Text Cleaning Pipeline

We implemented four distinct cleaning pipelines optimized for different use cases:

### 2.1 Basic Cleaning (`text_basic`)

**Target Use Cases:** Word2Vec, GloVe, FastText embeddings

**Steps Applied:**
1. Handle NaN/None values → Empty string
2. Remove URLs (`http://`, `https://`, `www.`)
3. Remove Twitter mentions (`@username`)
4. Remove hashtag symbols (`#` → keep text)
5. Convert to lowercase
6. Reduce repeated characters (e.g., "goooood" → "good")
7. Remove special characters (keep only letters and spaces)
8. Remove extra whitespace
9. Remove short words (length < 2)

**Rationale:** Preserves word forms while removing Twitter-specific noise. Suitable for word embedding models that benefit from seeing complete words.

### 2.2 Full Cleaning with Lemmatization (`text_lemmatized`)

**Target Use Cases:** TF-IDF, Bag of Words, Classical ML (SVM, Naive Bayes, Random Forest)

**Steps Applied:**
1. All basic cleaning steps
2. Expand contractions (e.g., "don't" → "do not")
3. Remove stopwords (English)
4. Apply lemmatization (reduce words to base form)

**Rationale:** Reduces vocabulary size significantly, which is beneficial for traditional ML models. Lemmatization preserves word meaning better than stemming.

### 2.3 Minimal Cleaning (`text_minimal`)

**Target Use Cases:** BERT, RoBERTa, DistilBERT, and other Transformer models

**Steps Applied:**
1. Handle NaN/None values
2. Remove URLs
3. Remove extra whitespace

**Rationale:** Modern transformer models are pre-trained on natural text and have their own tokenizers. Excessive cleaning can remove important contextual information that these models utilize.

### 2.4 Stemmed Cleaning (`text_stemmed`)

**Target Use Cases:** Comparison experiments, baseline models

**Steps Applied:**
1. All basic cleaning steps
2. Expand contractions
3. Remove stopwords
4. Apply Porter Stemming (more aggressive than lemmatization)

**Rationale:** Provides a comparison point against lemmatization. Stemming is faster but less linguistically accurate.

---

## 3. Cleaning Functions Reference

### 3.1 URL Removal
```python
def remove_urls(text):
    """Remove URLs from text (http://, https://, www.)"""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_pattern, '', text)
    text = re.sub(r'www\.\S+', '', text)
    return text
```

### 3.2 Mention Removal
```python
def remove_mentions(text):
    """Remove Twitter mentions (@username)"""
    return re.sub(r'@\w+', '', text)
```

### 3.3 Hashtag Processing
```python
def remove_hashtags(text):
    """Remove hashtag symbols but keep the text"""
    return re.sub(r'#', '', text)
```

### 3.4 Contraction Expansion
```python
def expand_contractions(text):
    """Expand contractions (e.g., don't → do not)"""
    # Uses predefined CONTRACTIONS dictionary
```

**Contractions Handled:**
- `don't` → `do not`
- `can't` → `cannot`
- `won't` → `will not`
- `I'm` → `I am`
- `they're` → `they are`
- And 40+ more common contractions

### 3.5 Stopword Removal
```python
def remove_stopwords(text, custom_stopwords=None):
    """Remove stopwords from text using NLTK English stopwords"""
```

### 3.6 Lemmatization
```python
def lemmatize_text(text):
    """Lemmatize words using WordNetLemmatizer"""
    # Example: running → run, better → good
```

### 3.7 Porter Stemming
```python
def stem_text(text):
    """Apply Porter Stemming to text"""
    # Example: running → run, happiness → happi
```

### 3.8 Repeated Character Reduction
```python
def remove_repeated_characters(text):
    """Reduce repeated characters to max 2"""
    # Example: goooood → good, happyyy → happyy
```

---

## 4. Quality Assurance

### 4.1 Empty Text Handling

After cleaning, some texts may become empty (e.g., texts containing only URLs or special characters). Our pipeline:

1. Identifies empty texts in each cleaning version
2. Removes records where minimal cleaning (most permissive) produces empty text
3. Preserves sentiment distribution after filtering

### 4.2 Validation Checks

- Verified no null values in cleaned columns
- Confirmed sentiment label distribution remains balanced
- Compared word count distributions across cleaning methods
- Analyzed vocabulary statistics for each method

---

## 5. Output Files

### 5.1 Main Dataset
**File:** `data/processed/twitter_cleaned.csv`

| Column | Description |
|--------|-------------|
| `original_text` | Raw tweet text |
| `label` | Numeric sentiment (-1, 0, 1) |
| `text_basic` | Basic cleaned text |
| `text_lemmatized` | Lemmatized text |
| `text_minimal` | Minimally cleaned text |
| `text_stemmed` | Stemmed text |
| `label_name` | Sentiment name (negative, neutral, positive) |

### 5.2 Individual Cleaned Versions

| File | Use Case |
|------|----------|
| `twitter_basic.csv` | Word2Vec, GloVe, FastText |
| `twitter_lemmatized.csv` | TF-IDF, Classical ML |
| `twitter_minimal.csv` | BERT, Transformers |

---

## 6. Recommendations for Model Training

### 6.1 Word Embedding Models (Word2Vec, GloVe, FastText)

**Use:** `text_basic`

**Reason:** These models learn word representations from context. Keeping complete word forms allows better semantic learning.

### 6.2 TF-IDF / Bag of Words + Classical ML

**Use:** `text_lemmatized`

**Reason:** 
- Reduced vocabulary improves model efficiency
- Stopword removal focuses on content words
- Lemmatization groups word variants together

### 6.3 Transformer Models (BERT, RoBERTa, DistilBERT)

**Use:** `text_minimal`

**Reason:**
- Transformers have sophisticated tokenizers
- Pre-trained on natural text with punctuation
- Excessive cleaning removes useful signals

### 6.4 Comparison Experiments

**Use:** `text_stemmed` vs `text_lemmatized`

**Purpose:** Evaluate impact of stemming vs lemmatization on model performance.

---

## 7. Dependencies

```
pandas>=1.5.0
numpy>=1.21.0
nltk>=3.8.0
matplotlib>=3.5.0
seaborn>=0.12.0
tqdm>=4.64.0
```

### NLTK Resources Required
- `punkt` - Tokenization
- `stopwords` - Stopword list
- `wordnet` - Lemmatization
- `omw-1.4` - Open Multilingual Wordnet

---

## 8. Reproducibility

To reproduce the cleaning pipeline:

1. Run all cells in `notebooks/02.text_cleaning.ipynb`
2. Ensure NLTK resources are downloaded
3. Output files will be generated in `data/processed/`

### Version Control

- Input: `data/raw/Twitter_Data.csv`
- Output: `data/processed/twitter_*.csv`
- Notebook: `notebooks/02.text_cleaning.ipynb`

---

## 9. Appendix: Complete Contractions Dictionary

| Contraction | Expansion |
|-------------|-----------|
| ain't | am not |
| aren't | are not |
| can't | cannot |
| couldn't | could not |
| didn't | did not |
| doesn't | does not |
| don't | do not |
| hadn't | had not |
| hasn't | has not |
| haven't | have not |
| he'd | he would |
| he'll | he will |
| he's | he is |
| i'd | i would |
| i'll | i will |
| i'm | i am |
| i've | i have |
| isn't | is not |
| it's | it is |
| let's | let us |
| mustn't | must not |
| shan't | shall not |
| she'd | she would |
| she'll | she will |
| she's | she is |
| shouldn't | should not |
| that's | that is |
| there's | there is |
| they'd | they would |
| they'll | they will |
| they're | they are |
| they've | they have |
| wasn't | was not |
| we'd | we would |
| we'll | we will |
| we're | we are |
| we've | we have |
| weren't | were not |
| what's | what is |
| won't | will not |
| wouldn't | would not |
| you'd | you would |
| you'll | you will |
| you're | you are |
| you've | you have |

---

*Documentation  of text preprocessing phase.*
