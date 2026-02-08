# Step 4: Load and prepare dataset
print('Step 4: Loading dataset...')

import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('../data/raw/Twitter_Data.csv')

# Drop missing values
df = df.dropna(subset=['clean_text', 'category'])

# Convert text to string
df['clean_text'] = df['clean_text'].astype(str)

print(' Dataset loaded successfully!')
print(f'Dataset shape: {df.shape}')
print(f'Columns: {df.columns.tolist()}')
