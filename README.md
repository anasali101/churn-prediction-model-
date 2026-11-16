# Churn Prediction Model

An interactive machine learning web application that predicts customer churn using an XGBoost classification model.

The application provides a dashboard for analyzing customer churn, generating individual and batch predictions, and understanding the factors influencing model decisions.

## Live Demo

- **Frontend:** https://churn-prediction-model-4f4m.vercel.app/
- **Backend API:** https://churn-prediction-model-mnbw.onrender.com
- **API Documentation:** https://churn-prediction-model-mnbw.onrender.com/docs

## Features

- **Customer Churn Prediction:** Predicts the likelihood of customers leaving a service using an XGBoost model.
- **Interactive Dashboard:** Explore prediction results and customer churn information.
- **CSV Batch Prediction:** Upload customer datasets and generate predictions for multiple records.
- **Model Explainability:** Uses SHAP and LIME to help interpret machine learning predictions.
- **PDF Export:** Export prediction results as PDF reports.
- **REST API:** FastAPI backend connects the machine learning model with the frontend.

## Technology Stack

### Frontend
- React
- TypeScript
- Vite

### Backend
- Python
- FastAPI
- Uvicorn

### Machine Learning and Data Processing
- XGBoost
- Scikit-learn
- Pandas
- NumPy
- SHAP
- LIME

## Architecture

```text
        React + TypeScript
               |
               | HTTP Requests
               v
          FastAPI Backend
               |
               v
       Data Preprocessing
               |
               v
        XGBoost Model
               |
               v
       Churn Predictions
               |
               v
      Dashboard and Reports
```

## Run Locally

### Prerequisites

- Python 3.10+
- Node.js and npm
- Git

### 1. Clone the repository

```bash
git clone https://github.com/anasali101/churn-prediction-model-.git
cd churn-prediction-model-
```

### 2. Start the backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend API: http://localhost:8000

API documentation: http://localhost:8000/docs

### 3. Start the frontend

Open another terminal:

```bash
cd frontend
npm install
```

Create a `.env` file inside the frontend directory:

```env
VITE_API_URL=http://localhost:8000
```

Start the development server:

```bash
npm run dev
```

Frontend: http://localhost:5173

## Deployment

The application is deployed using:

- **Vercel** – Frontend hosting.
- **Render** – Backend API hosting.

The frontend communicates with the backend through the `VITE_API_URL` environment variable.

## Project Purpose

Customer churn can affect business growth and customer retention. This project explores how machine learning can help identify customers who may be likely to discontinue a service.

It combines machine learning, data processing, API development, and frontend engineering into one web application.

## Disclaimer

This project is intended for educational and demonstration purposes. Predictions are model estimates and should not be treated as guaranteed outcomes or used as the sole basis for business decisions.

## Author

**MD Anas Ali**

- GitHub: https://github.com/anasali101
- LinkedIn: https://linkedin.com/in/md-anas-ali-b99034230

---

Built as a machine learning and full-stack development project.
