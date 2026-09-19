import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# ============================================
# 1. LOAD & FEATURE ENGINEERING
# ============================================
df = pd.read_csv("data/cleaned_transactions.csv")

# Balance mismatch features (already have errorBalanceOrig)
df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']

# Full balance wipe (attacker took everything)
df['fullBalanceWipe'] = ((df['newbalanceOrig'] == 0) & (df['oldbalanceOrg'] > 0)).astype(int)

# Amount relative to sender's balance (fraud often takes a large % of balance)
df['amountToBalanceRatio'] = df['amount'] / (df['oldbalanceOrg'] + 1)  # +1 avoids divide-by-zero

# Time of day (step is hours since simulation start; 24-hour cycle)
df['hourOfDay'] = df['step'] % 24

# Destination account had zero balance before receiving (common in fraud - "mule" accounts)
df['destWasEmpty'] = (df['oldbalanceDest'] == 0).astype(int)

features = [
    'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'errorBalanceOrig', 'errorBalanceDest',
    'fullBalanceWipe', 'amountToBalanceRatio',
    'hourOfDay', 'destWasEmpty'
]

X = df[features]
y = df['isFraud']

# ============================================
# 2. ISOLATION FOREST (Unsupervised)
# ============================================
print("=" * 50)
print("ISOLATION FOREST (Unsupervised)")
print("=" * 50)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

actual_fraud_rate = y.mean()  # use real data-driven contamination, not a guess

iso_model = IsolationForest(
    n_estimators=200,
    contamination=actual_fraud_rate,
    max_samples='auto',
    random_state=42,
    n_jobs=-1  # use all CPU cores, faster on large data
)

df['anomaly'] = iso_model.fit_predict(X_scaled)
df['anomaly_flag'] = (df['anomaly'] == -1).astype(int)

print(classification_report(y, df['anomaly_flag'], target_names=['Normal', 'Fraud']))
print("ROC-AUC:", roc_auc_score(y, -iso_model.decision_function(X_scaled)))

# ============================================
# 3. RANDOM FOREST (Supervised)
# ============================================
print("\n" + "=" * 50)
print("RANDOM FOREST (Supervised)")
print("=" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify keeps fraud ratio same in both sets
)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight='balanced',  # critical for imbalanced data - tells model fraud is rare but important
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

# Feature importance - which features matter most for detecting fraud
importance = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature Importance:\n", importance)

# ============================================
# 4. SAVE MODELS FOR API USE LATER
# ============================================
import joblib
joblib.dump(rf_model, "app/models_store/random_forest_model.pkl")
joblib.dump(scaler, "app/models_store/scaler.pkl")
print("\nModels saved!")