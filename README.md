# Church Prediction

A machine learning-powered customer churn prediction dashboard.

## Overview

Church Prediction is a web application designed to analyze customer information, estimate churn risk, evaluate model performance, and generate customer retention recommendations.

## Features

- Customer churn prediction
- Dataset overview and analytics
- Model evaluation
- Individual customer risk analysis
- CSV-based bulk churn prediction (currently being fixed)
- Customer retention recommendations
- Risk classification: Low, Medium, and High

## Tech Stack

### Frontend
- React
- TypeScript
- Vite
- Recharts
- Axios

### Backend
- Python
- FastAPI
- Pandas
- Scikit-learn
- XGBoost
- SHAP
- LIME

## Project Structure

```text
hacathon/
├── backend/
│   ├── app/
│   ├── data/
│   ├── models/
│   ├── requirements.txt
│   └── train_model.py
├── frontend/
│   ├── public/
│   ├── src/
│   └── package.json
├── .gitignore
└── README.md
```

## Local Setup

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### Frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend: http://localhost:5173

Backend: http://localhost:8000

API documentation: http://localhost:8000/docs

## Project Status

This project is under development. Some features may need further testing and improvements.

## Disclaimer

This project uses a synthetic customer dataset for demonstration and educational purposes. Predictions should not be treated as guaranteed outcomes.
