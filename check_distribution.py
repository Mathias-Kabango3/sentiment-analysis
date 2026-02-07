import pandas as pd

# Read more data to check distribution
df = pd.read_csv('data/raw/Twitter_Data.csv')
print('=== DATASET SUMMARY ===')
print('Total tweets:', len(df))
print('\n=== SENTIMENT DISTRIBUTION ===')
print(df['category'].value_counts().sort_index())
print('\n=== SENTIMENT MAPPING ===')
print('-1 = Negative')
print(' 0 = Neutral')
print(' 1 = Positive')
