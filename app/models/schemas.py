from pydantic import BaseModel

class TransactionInput(BaseModel):
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    step: int

class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    risk_level: str