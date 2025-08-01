import pandas as pd
import numpy as np

# Read the final merged dataset
df = pd.read_csv('dataset/preprocessed_data_final.csv')

print('Original dataset:')
print(f'Total samples: {len(df)}')
print(df['label'].value_counts())
print('Distribution:')
print(df['label'].value_counts(normalize=True) * 100)

# Separate the classes
sarcastic_samples = df[df['label'] == 1]  # minority class
non_sarcastic_samples = df[df['label'] == 0]  # majority class

print(f'\nSarcastic samples: {len(sarcastic_samples)}')
print(f'Non-sarcastic samples: {len(non_sarcastic_samples)}')

# Undersample the majority class by removing 40% (keep 60% of original majority class)
np.random.seed(42)  # for reproducibility
target_majority_size = int(len(non_sarcastic_samples) * 0.6)  # Keep 60% of majority class (remove 40%)
undersampled_non_sarcastic = non_sarcastic_samples.sample(n=target_majority_size, random_state=42)

print(f'\nAfter removing 40% of majority class:')
print(f'Original majority class size: {len(non_sarcastic_samples)}')
print(f'New majority class size: {len(undersampled_non_sarcastic)} (removed 40%, kept 60%)')

# Combine the balanced dataset
balanced_df = pd.concat([sarcastic_samples, undersampled_non_sarcastic], ignore_index=True)

# Shuffle the dataset
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f'\nDataset after undersampling:')
print(f'Total samples: {len(balanced_df)}')
print(balanced_df['label'].value_counts())
print('Distribution:')
print(balanced_df['label'].value_counts(normalize=True) * 100)

# Data cleaning operations
print('\n--- Starting data cleaning ---')

# Check for duplicate rows
duplicate_rows = balanced_df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_rows}")

# Remove duplicate rows (if any)
df_no_duplicates = balanced_df.drop_duplicates()

# Remove the 'Unnamed: 0' column if it exists
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

# Display the head of the cleaned DataFrame
print('\nFirst 5 rows of cleaned dataset:')
print(df_cleaned.head())

# Save the final cleaned and balanced dataset
df_cleaned.to_csv('dataset/processed_dataset/cleaned_dataset.csv', index=False)

print('\nSuccessfully created balanced and cleaned dataset saved as dataset/cleaned_dataset.csv')