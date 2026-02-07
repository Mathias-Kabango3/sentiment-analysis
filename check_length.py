import pandas as pd
import numpy as np

df = pd.read_csv('data/raw/Twitter_Data.csv')
df['text_length'] = df['clean_text'].apply(len)
df['word_count'] = df['clean_text'].apply(lambda x: len(str(x).split()))

print('=== TEXT LENGTH STATISTICS ===')
print('Average characters:', df['text_length'].mean())
print('Max characters:', df['text_length'].max())
print('Min characters:', df['text_length'].min())
print('\nAverage words:', df['word_count'].mean())
print('Max words:', df['word_count'].max())
print('Min words:', df['word_count'].min())
