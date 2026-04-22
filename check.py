import pandas as pd

df = pd.read_csv('data/raw/landmarks_sequences.csv')

# Drop all four and three rows
df_cleaned = df[~df['label'].isin(['four', 'three'])]
df_cleaned.to_csv('data/raw/landmarks_sequences.csv', index=False)

print(df_cleaned['label'].value_counts())