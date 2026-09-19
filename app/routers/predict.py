from fastapi import APIRouter
import joblib
import pandas as pd
from app.models.schemas import TransactionInput, PredictionResponse

router = APIRouter()

# Model aur scaler load karo (ek baar, jab server start ho)
model = joblib.load("app/models_store/random_forest_model.pkl")

@router.post("/predict", response_model=PredictionResponse)
def predict_fraud(transaction: TransactionInput):
    # Same features banao jo training ke waqt banaye the
    errorBalanceOrig = transaction.newbalanceOrig + transaction.amount - transaction.oldbalanceOrg
    errorBalanceDest = transaction.oldbalanceDest + transaction.amount - transaction.newbalanceDest
    fullBalanceWipe = int(transaction.newbalanceOrig == 0 and transaction.oldbalanceOrg > 0)
    amountToBalanceRatio = transaction.amount / (transaction.oldbalanceOrg + 1)
    hourOfDay = transaction.step % 24
    destWasEmpty = int(transaction.oldbalanceDest == 0)

    # DataFrame banao (training ke time jaisa hi order mein)
    input_data = pd.DataFrame([{
        'amount': transaction.amount,
        'oldbalanceOrg': transaction.oldbalanceOrg,
        'newbalanceOrig': transaction.newbalanceOrig,
        'errorBalanceOrig': errorBalanceOrig,
        'errorBalanceDest': errorBalanceDest,
        'fullBalanceWipe': fullBalanceWipe,
        'amountToBalanceRatio': amountToBalanceRatio,
        'hourOfDay': hourOfDay,
        'destWasEmpty': destWasEmpty
    }])

    # Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]  # fraud hone ka probability

    # Risk level decide karo
    if probability > 0.7:
        risk = "High"
    elif probability > 0.3:
        risk = "Medium"
    else:
        risk = "Low"

    return PredictionResponse(
        is_fraud=bool(prediction),
        fraud_probability=round(float(probability), 4),
        risk_level=risk
    )