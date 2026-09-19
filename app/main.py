from fastapi import FastAPI

app = FastAPI(title="Finance Anomaly Detector")

@app.get("/")
def read_root():
    return {"message": "Finance Anomaly Detector"}