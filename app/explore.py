import pandas as pd

df = pd.read_csv("data/PS_20174392719_1491204439457_log.csv")

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head())
print("\nData types:\n", df.dtypes)
# Fraud distribution
print("\nFraud count:\n", df['isFraud'].value_counts())
print("\nFraud percentage:", (df['isFraud'].sum() / len(df)) * 100, "%")

# Transaction types
print("\nTransaction types:\n", df['type'].value_counts())

# Fraud by transaction type
print("\nFraud by type:\n", df.groupby('type')['isFraud'].sum())

# Missing values check
print("\nMissing values:\n", df.isnull().sum())