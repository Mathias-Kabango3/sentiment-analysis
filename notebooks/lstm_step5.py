# Step 5: Prepare data for LSTM
print('Step 5: Preparing data for LSTM...')

import pandas as pd
import numpy as np

# Load dataset (again for this step)
df = pd.read_csv('../data/raw/Twitter_Data.csv')
df = df.dropna(subset=['clean_text', 'category'])
df['clean_text'] = df['clean_text'].astype(str)

# Map sentiment categories (-1, 0, 1) to (0, 1, 2)
df['sentiment'] = df['category'].map({-1: 0, 0: 1, 1: 2})

# Prepare data
X = df['clean_text'].values
y = df['sentiment'].values

print(f'X shape: {X.shape}')
print(f'y shape: {y.shape}')
print('\nSentiment distribution:')
print(df['sentiment'].value_counts().sort_index())
print('\nFirst 2 samples:')
for i in range(2):
    print(f'{i+1}. Text: {X[i][:80]}...')
    print(f'   Sentiment: {y[i]} (original: {df["category"].iloc[i]})')
