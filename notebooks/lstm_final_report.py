# LSTM Final Results Report
# Comparative Analysis of Embedding Methods

import pandas as pd

print('=== LSTM WITH DIFFERENT EMBEDDINGS - RESULTS ===')
print('Twitter Sentiment Analysis')
print('=' * 60)

results = {
    'Embedding Method': ['Random Embeddings', 'Word2Vec Skip-gram', 'Word2Vec CBOW'],
    'Accuracy': [0.9690, 0.9195, 0.9158],
    'Loss': [0.1158, 0.2558, 0.2641],
    'Training Time (min)': [26.9, 25.9, 24.7],
    'Best F1-Score': ['Neutral (0.98)', 'Neutral (0.94)', 'Neutral (0.94)'],
    'Worst F1-Score': ['Negative (0.94)', 'Negative (0.86)', 'Negative (0.84)']
}

df = pd.DataFrame(results)
print(df.to_string(index=False))

print('\n=== KEY INSIGHTS ===')
print('1. Random embeddings achieved the highest accuracy (96.9%)')
print('2. Word2Vec embeddings performed similarly (91.6-91.9%)')
print('3. Skip-gram slightly outperformed CBOW')
print('4. Negative sentiment was hardest to classify across all methods')
print('5. Training times were comparable (~25 minutes each)')

print('\n=== CONCLUSION ===')
print('Task-specific random embeddings outperformed general-purpose')
print('Word2Vec embeddings for this sentiment analysis task.')
