import pandas as pd

# Load full data
df = pd.read_csv("data/PS_20174392719_1491204439457_log.csv")

# Filter: sirf CASH_OUT aur TRANSFER lo (kyunki fraud sirf inhi mein hota hai)
df_filtered = df[df['type'].isin(['CASH_OUT', 'TRANSFER'])].copy()

print("Original shape:", df.shape)
print("Filtered shape:", df_filtered.shape)
print("\nFraud count in filtered data:\n", df_filtered['isFraud'].value_counts())

# Feature engineering — naye useful columns banate hain
df_filtered['balanceDiffOrig'] = df_filtered['oldbalanceOrg'] - df_filtered['newbalanceOrig']
df_filtered['balanceDiffDest'] = df_filtered['newbalanceDest'] - df_filtered['oldbalanceDest']
df_filtered['errorBalanceOrig'] = df_filtered['newbalanceOrig'] + df_filtered['amount'] - df_filtered['oldbalanceOrg']

print("\nNew columns preview:\n", df_filtered[['amount', 'balanceDiffOrig', 'balanceDiffDest', 'errorBalanceOrig']].head())

# Save cleaned data taaki baar baar poora dataset load na karna pade
df_filtered.to_csv("data/cleaned_transactions.csv", index=False)
print("\nCleaned data saved!")