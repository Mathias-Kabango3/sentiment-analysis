import pandas as pd
print('Pandas version:', pd.__version__)

# Try reading with error details
try:
    df = pd.read_csv('data/raw/Twitter_Data.csv', nrows=3, on_bad_lines='skip')
    print('SUCCESS!')
    print('Columns:', list(df.columns))
    print('First row values:')
    for col in df.columns:
        print(f'  {col}: {df[col].iloc[0]}')
except Exception as e:
    print('ERROR:', str(e))
    import traceback
    traceback.print_exc()
