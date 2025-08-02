import pandas as pd

df = pd.read_csv('dataset/processed_dataset/cleaned_dataset.csv')

print("Class distribution:")
print(df['label'].value_counts())
print(f"\nTotal samples: {len(df)}")

class_counts = df['label'].value_counts()
minority_class = class_counts.idxmin()
majority_class = class_counts.idxmax()

print(f"\nMinority class: {minority_class} ({class_counts[minority_class]} samples)")
print(f"Majority class: {majority_class} ({class_counts[majority_class]} samples)")