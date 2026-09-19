# Finance Anomaly Detector

An AI-powered financial fraud detection system built with FastAPI, scikit-learn, and machine learning. This project detects fraudulent transactions in real-time using both unsupervised (Isolation Forest) and supervised (Random Forest) machine learning approaches.

## Features

- **Real-time fraud prediction** via REST API
- **Dual ML approach**: Isolation Forest (unsupervised anomaly detection) + Random Forest (supervised classification)
- **Feature engineering**: Balance mismatch detection, transaction ratio analysis, time-based patterns
- **97% precision, 100% recall** on fraud detection using Random Forest
- **Interactive API documentation** via FastAPI's built-in Swagger UI

## Tech Stack

- **Backend**: FastAPI, Uvicorn
- **Data Analytics**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (Isolation Forest, Random Forest)
- **Model Persistence**: Joblib

## Dataset

Built using the [Synthetic Financial Datasets For Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1) (PaySim) dataset — 6.3M+ simulated financial transactions.

## Project Architecture

```
app/
  main.py              # FastAPI app entry point
  models/              # Pydantic request/response schemas
  routers/             # API endpoint definitions
  services/            # Business logic (future expansion)
  models_store/        # Trained ML models (.pkl files)
  clean_data.py        # Data cleaning and feature engineering
  model.py             # Model training script
```

## Key Insights from Analysis

- Fraud occurs exclusively in `CASH_OUT` and `TRANSFER` transaction types
- Fraud rate is extremely low (~0.3%), requiring class-imbalance-aware techniques
- Top fraud predictors: balance mismatch (`errorBalanceOrig`) and amount-to-balance ratio

## How to Run Locally

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Run the server: `uvicorn app.main:app --reload`
6. Visit `http://127.0.0.1:8000/docs` for interactive API documentation

## API Endpoints

- `GET /` — Health check
- `POST /api/predict` — Predict whether a transaction is fraudulent

## Future Improvements

- AI-generated natural language explanations for flagged transactions (Claude API integration)
- Frontend dashboard for visualizing anomalies and trends
- Database integration for persistent transaction storage
- Deployment to cloud platform

## Author

Bhanu — [GitHub](https://github.com/bhanuduhoon77)