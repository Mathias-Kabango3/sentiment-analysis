import pandas as pd
import numpy as np

df = pd.read_csv('data/raw/Twitter_Data.csv')

# Check for missing values
print('=== MISSING VALUES CHECK ===')
print(df.isnull().sum())

# Drop rows with missing text
df_clean = df.dropna(subset=['clean_text'])
print(f'\nRows before cleaning: {len(df)}')
print(f'Rows after cleaning: {len(df_clean)}')

# Convert text to string
df_clean['clean_text'] = df_clean['clean_text'].astype(str)

# Calculate lengths
df_clean['text_length'] = df_clean['clean_text'].apply(len)
df_clean['word_count'] = df_clean['clean_text'].apply(lambda x: len(x.split()))

print('\n=== TEXT LENGTH STATISTICS ===')
print('Average characters:', df_clean['text_length'].mean())
print('Max characters:', df_clean['text_length'].max())
print('Min characters:', df_clean['text_length'].min())
print('\nAverage words:', df_clean['word_count'].mean())
print('Max words:', df_clean['word_count'].max())
print('Min words:', df_clean['word_count'].min())
