import pandas as pd
import numpy as np

df = pd.read_csv('dataset/preprocessed_data_final.csv')

print('Original dataset:')
print(f'Total samples: {len(df)}')
print(df['label'].value_counts())
print('Distribution:')
print(df['label'].value_counts(normalize=True) * 100)

sarcastic_samples = df[df['label'] == 1]
non_sarcastic_samples = df[df['label'] == 0]

print(f'\nSarcastic samples: {len(sarcastic_samples)}')
print(f'Non-sarcastic samples: {len(non_sarcastic_samples)}')

np.random.seed(42)
target_majority_size = int(len(non_sarcastic_samples) * 0.6)
undersampled_non_sarcastic = non_sarcastic_samples.sample(n=target_majority_size, random_state=42)

print(f'\nAfter removing 40% of majority class:')
print(f'Original majority class size: {len(non_sarcastic_samples)}')
print(f'New majority class size: {len(undersampled_non_sarcastic)} (removed 40%, kept 60%)')

balanced_df = pd.concat([sarcastic_samples, undersampled_non_sarcastic], ignore_index=True)

balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f'\nDataset after undersampling:')
print(f'Total samples: {len(balanced_df)}')
print(balanced_df['label'].value_counts())
print('Distribution:')
print(balanced_df['label'].value_counts(normalize=True) * 100)

print('\n--- Starting data cleaning ---')

duplicate_rows = balanced_df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_rows}")

df_no_duplicates = balanced_df.drop_duplicates()

if 'Unnamed: 0' in df_no_duplicates.columns:
    df_cleaned = df_no_duplicates.drop(columns=['Unnamed: 0'])
    print("'Unnamed: 0' column dropped.")
else:
    df_cleaned = df_no_duplicates
    print("'Unnamed: 0' column not found.")

print(f'\nFinal cleaned dataset:')
print(f'Total samples: {len(df_cleaned)}')
print(df_cleaned['label'].value_counts())
print('Distribution:')
print(df_cleaned['label'].value_counts(normalize=True) * 100)

print('\nFirst 5 rows of cleaned dataset:')
print(df_cleaned.head())

df_cleaned.to_csv('dataset/processed_dataset/cleaned_dataset.csv', index=False)

print('\nSuccessfully created balanced and cleaned dataset saved as dataset/processed_dataset/cleaned_dataset.csv')