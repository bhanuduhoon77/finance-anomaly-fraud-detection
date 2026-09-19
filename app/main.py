from fastapi import FastAPI
from app.routers import predict

app = FastAPI(title="Finance Anomaly Detector")

app.include_router(predict.router, prefix="/api", tags=["Prediction"])

@app.get("/")
def read_root():
    return {"message": "Finance Anomaly Detector API is running"}