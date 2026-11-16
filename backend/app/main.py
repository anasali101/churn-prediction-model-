from pathlib import Path
import io
import os
from typing import Any

from .retention import generate_retention_plan

import numpy as np
import pandas as pd
import shap

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from pydantic import BaseModel, Field

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    roc_curve,
    confusion_matrix,
)

from xgboost import XGBClassifier
from lime.lime_tabular import LimeTabularExplainer


# Paths relative to backend/app/main.py
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.json"
DATA_PATH = BASE_DIR / "data" / "churn_synthetic.csv"

FEATURE_COLUMNS = [
    "tenure",
    "monthly_charges",
    "contract_monthly",
    "contract_one_year",
    "contract_two_year",
    "internet_fiber",
    "tech_support",
    "paperless_billing",
    "senior_citizen",
    "dependents",
]

app = FastAPI(
    title="ChurnIQ API",
    description="XGBoost churn predictions with SHAP and LIME explanations",
    version="2.0.0",
)

# Configure allowed frontend origins. In production, set FRONTEND_ORIGINS
# to your actual HTTPS frontend origin(s), comma-separated.
_frontend_origins = os.getenv(
    "FRONTEND_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
)
ALLOWED_ORIGINS = [
    origin.strip().rstrip("/")
    for origin in _frontend_origins.split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# Enable only when HTTPS is configured directly in Uvicorn or correctly at
# a reverse proxy. Leave disabled for local HTTP development.
if os.getenv("ENABLE_HTTPS_REDIRECT", "false").strip().lower() in {"1", "true", "yes"}:
    app.add_middleware(HTTPSRedirectMiddleware)


@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"

    # HSTS must only be sent when clients access the API over HTTPS.
    if request.url.scheme == "https":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response



class Customer(BaseModel):
    tenure: int = Field(ge=0, le=120)
    monthly_charges: float = Field(ge=0)
    contract: str = "Monthly"
    internet_fiber: bool = False
    tech_support: bool = False
    paperless_billing: bool = True
    senior_citizen: bool = False
    dependents: bool = False


def customer_features(customer: Customer) -> pd.DataFrame:
    contract = customer.contract.strip().lower()

    contract_values = {
        "monthly": (1, 0, 0),
        "one year": (0, 1, 0),
        "one-year": (0, 1, 0),
        "annual": (0, 1, 0),
        "two year": (0, 0, 1),
        "two-year": (0, 0, 1),
        "two years": (0, 0, 1),
    }

    if contract not in contract_values:
        raise HTTPException(
            status_code=422,
            detail="Contract must be Monthly, One year, or Two year.",
        )

    monthly, one_year, two_year = contract_values[contract]

    row = {
        "tenure": customer.tenure,
        "monthly_charges": customer.monthly_charges,
        "contract_monthly": monthly,
        "contract_one_year": one_year,
        "contract_two_year": two_year,
        "internet_fiber": int(customer.internet_fiber),
        "tech_support": int(customer.tech_support),
        "paperless_billing": int(customer.paperless_billing),
        "senior_citizen": int(customer.senior_citizen),
        "dependents": int(customer.dependents),
    }

    return pd.DataFrame(
        [[row[name] for name in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
        dtype=float,
    )


# Load the saved trained model and background data.
if not MODEL_PATH.exists():
    raise RuntimeError(f"Trained model not found: {MODEL_PATH}")

if not DATA_PATH.exists():
    raise RuntimeError(f"Training dataset not found: {DATA_PATH}")

model = XGBClassifier()
model.load_model(str(MODEL_PATH))

training_df = pd.read_csv(DATA_PATH)

print("Dataset columns:", training_df.columns.tolist())
print(training_df.head(3).to_string())

missing = set(FEATURE_COLUMNS) - set(training_df.columns)

if missing:
    raise RuntimeError(
        f"Dataset is missing required feature columns: {sorted(missing)}"
    )

background = training_df[FEATURE_COLUMNS].astype(float)

# SHAP explains this model's predictions.
shap_explainer = shap.TreeExplainer(model)

# LIME uses training examples as its reference distribution.
lime_explainer = LimeTabularExplainer(
    training_data=background.to_numpy(),
    feature_names=FEATURE_COLUMNS,
    class_names=["No churn", "Churn"],
    mode="classification",
    discretize_continuous=True,
    random_state=42,
)


def lime_predict_proba(values):
    frame = pd.DataFrame(values, columns=FEATURE_COLUMNS)
    return model.predict_proba(frame)


@app.get("/")
def home():
    return {
        "message": "ChurnIQ API is running",
        "model": "XGBoost",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": "xgboost",
    }

@app.get("/dataset-info")
def dataset_info():
    return {
        "columns": training_df.columns.tolist(),
        "total_rows": len(training_df),
        "possible_target_columns": [
            col for col in training_df.columns
            if col.lower() in [
                "churn", "exited", "target", "churn_label"
            ]
        ],
    }

@app.get("/evaluate")
def evaluate_model():
    try:
        # Read the dataset
        df = pd.read_csv(DATA_PATH)

        # Confirm required columns exist
        required_columns = FEATURE_COLUMNS + ["churn"]
        missing = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Dataset is missing columns: {missing}",
            )

        # Prepare input features and target
        X = df[FEATURE_COLUMNS].astype(float)
        y = df["churn"]

        # Convert common text labels to 0 and 1 if necessary
        if not pd.api.types.is_numeric_dtype(y):
            label_map = {
                "no": 0,
                "yes": 1,
                "false": 0,
                "true": 1,
                "0": 0,
                "1": 1,
                "no churn": 0,
                "churn": 1,
            }

            y = y.astype(str).str.strip().str.lower().map(label_map)

        if y.isna().any() or not set(y.unique()).issubset({0, 1}):
            raise HTTPException(
                status_code=400,
                detail="The churn column must contain binary 0/1 "
                       "or recognized Yes/No labels.",
            )

        y = y.astype(int)

        if y.nunique() != 2:
            raise HTTPException(
                status_code=400,
                detail="ROC-AUC requires both churn classes.",
            )

        # Hold out customers for an independent evaluation split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )

        # Train an evaluation model using the same feature columns
        evaluation_model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        )

        evaluation_model.fit(X_train, y_train)

        # Evaluate on customers not used to fit this evaluation model
        probabilities = evaluation_model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        # Calculate ROC curve points
        false_positive_rates, true_positive_rates, thresholds = roc_curve(
            y_test, probabilities
        )

        roc_curve_data = [
            {
                "fpr": round(float(fpr), 6),
                "tpr": round(float(tpr), 6),
            }
            for fpr, tpr in zip(
                false_positive_rates, true_positive_rates
            )
        ]

        # Calculate confusion matrix values
        tn, fp, fn, tp = confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).ravel()

        return {
            "evaluation_type": "holdout_split",
            "test_customers": int(len(X_test)),
            "roc_auc": round(
                float(roc_auc_score(y_test, probabilities)), 4
            ),
            "accuracy": round(
                float(accuracy_score(y_test, predictions)), 4
            ),
            "churn_rate_percent": round(
                float(y_test.mean() * 100), 2
            ),
            "roc_curve": roc_curve_data,
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
                "labels": ["No churn", "Churn"],
            },
            "note": (
                "Metrics are from a newly trained evaluation model, "
                "not necessarily the saved production model."
            ),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Model evaluation failed: {str(exc)}",
        ) from exc

@app.post("/predict")
def predict(customer: Customer):
    try:
        features = customer_features(customer)

        # Real model prediction
        probability = float(model.predict_proba(features)[0, 1])
        risk_percent = round(probability * 100, 2)

        if risk_percent >= 70:
            risk = "High"
            recommendation = (
                "Consider proactive retention outreach and investigate "
                "service or support issues."
            )
        elif risk_percent >= 40:
            risk = "Medium"
            recommendation = (
                "Review customer engagement, service satisfaction, "
                "and available contract options."
            )
        else:
            risk = "Low"
            recommendation = (
                "Continue regular engagement and monitor customer activity."
            )

                # SHAP feature contributions for this customer
        shap_result = shap_explainer(features)

        shap_values = np.asarray(shap_result.values)

        # Handle SHAP versions that return separate class outputs.
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        contributions = shap_values.reshape(-1)

        # Get the baseline for the output being explained.
        base_values = np.asarray(shap_result.base_values)

        if base_values.ndim == 0:
            base_value = float(base_values)
        elif base_values.ndim == 1:
            base_value = float(base_values[0])
        elif base_values.ndim == 2:
            class_index = (
                1 if base_values.shape[1] > 1 else 0
            )
            base_value = float(base_values[0, class_index])
        else:
            raise ValueError(
                f"Unexpected SHAP base-value shape: {base_values.shape}"
            )

        # SHAP's additive result in the model's explained output space.
        shap_output_value = float(
            base_value + contributions.sum()
        )

        # Feature contributions ordered by absolute impact.
        shap_explanation = [
            {
                "feature": feature,
                "value": float(features.iloc[0][feature]),
                "impact": round(float(contributions[index]), 6),
            }
            for index, feature in enumerate(FEATURE_COLUMNS)
        ]
        shap_explanation.sort(
            key=lambda item: abs(item["impact"]),
            reverse=True,
        )

        # LIME explanation for this individual customer
        lime_result = lime_explainer.explain_instance(
            data_row=features.iloc[0].to_numpy(),
            predict_fn=lime_predict_proba,
            labels=(1,),
            num_features=len(FEATURE_COLUMNS),
            num_samples=1000,
        )

        lime_explanation = [
            {
                "feature_condition": condition,
                "impact": round(float(weight), 6),
            }
            for condition, weight in lime_result.as_list(label=1)
        ]

        return {
            "churn_risk_percent": risk_percent,
            "risk_segment": risk,
            "recommendation": recommendation,
            "model_type": "xgboost",
            "explanation_method": ["SHAP", "LIME"],
            "lime_explanation": lime_explanation,
            "customer_features": features.iloc[0].to_dict(),
                        "shap_explanation": shap_explanation,
            "shap_base_value": base_value,
            "shap_output_value": shap_output_value,
            "shap_output_scale": "raw_model_output",
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction or explanation failed: {str(exc)}",
        ) from exc


@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)):
    """Predict churn risk for customers in an uploaded CSV file."""

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    max_csv_bytes = 5 * 1024 * 1024

    try:
        contents = await file.read(max_csv_bytes + 1)

        if not contents:
            raise HTTPException(status_code=400, detail="The uploaded CSV file is empty.")

        if len(contents) > max_csv_bytes:
            raise HTTPException(status_code=413, detail="CSV file exceeds the 5 MB upload limit.")

        try:
            df = pd.read_csv(io.BytesIO(contents))
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            raise HTTPException(
                status_code=400,
                detail="Could not read the CSV. Check that it is a valid UTF-8 CSV file.",
            ) from exc
        finally:
            await file.close()

        required = {"tenure", "monthly_charges"}
        missing = required - set(df.columns)

        contract_columns = {
            "contract_monthly",
            "contract_one_year",
            "contract_two_year",
        }

        has_text_contract = "contract" in df.columns
        has_one_hot_contract = contract_columns.issubset(df.columns)

        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Missing CSV columns: {sorted(missing)}",
            )

        if not has_text_contract and not has_one_hot_contract:
            raise HTTPException(
                status_code=422,
                detail=(
                    "CSV must contain either a 'contract' column or all three "
                    "columns: contract_monthly, contract_one_year, contract_two_year."
                ),
            )

        if df.empty:
            raise HTTPException(status_code=400, detail="The uploaded CSV contains no customers.")

        if len(df) > 10000:
            raise HTTPException(status_code=413, detail="Upload a maximum of 10,000 customers at a time.")

        results = []

        def as_bool(value, default=False):
            if pd.isna(value):
                return default
            if isinstance(value, (bool, np.bool_)):
                return bool(value)
            if value in (1, 0):
                return bool(value)

            normalized = str(value).strip().lower()

            if normalized in ("true", "yes", "1"):
                return True
            if normalized in ("false", "no", "0"):
                return False

            raise ValueError(
                f"Invalid boolean value '{value}'. Use true/false, yes/no, or 1/0."
            )

        def contract_from_row(row):
            if has_text_contract:
                return str(row["contract"])

            values = [
                as_bool(row["contract_monthly"]),
                as_bool(row["contract_one_year"]),
                as_bool(row["contract_two_year"]),
            ]

            if sum(values) != 1:
                raise ValueError(
                    "Exactly one contract column must be 1: "
                    "contract_monthly, contract_one_year, contract_two_year."
                )

            if values[0]:
                return "Monthly"
            if values[1]:
                return "One year"
            return "Two year"

        for index, row in df.iterrows():
            try:
                customer = Customer(
                    tenure=int(row["tenure"]),
                    monthly_charges=float(row["monthly_charges"]),
                    contract=contract_from_row(row),
                    internet_fiber=as_bool(row.get("internet_fiber", False)),
                    tech_support=as_bool(row.get("tech_support", False)),
                    paperless_billing=as_bool(
                        row.get("paperless_billing", True), default=True
                    ),
                    senior_citizen=as_bool(row.get("senior_citizen", False)),
                    dependents=as_bool(row.get("dependents", False)),
                )

                features = customer_features(customer)
                probability = float(model.predict_proba(features)[0, 1])
                risk_percent = round(probability * 100, 2)

                if risk_percent >= 70:
                    segment = "High"
                elif risk_percent >= 40:
                    segment = "Medium"
                else:
                    segment = "Low"

                results.append({
                    "customer_index": int(index),
                    "churn_risk_percent": risk_percent,
                    "risk_segment": segment,
                })

            except (ValueError, TypeError, OverflowError) as exc:
                raise ValueError(f"CSV row {index + 2}: {exc}") from exc

        result_df = pd.DataFrame(results)

        return {
            "total_customers": len(results),
            "high_risk": int((result_df["risk_segment"] == "High").sum()),
            "medium_risk": int((result_df["risk_segment"] == "Medium").sum()),
            "low_risk": int((result_df["risk_segment"] == "Low").sum()),
            "customers": results,
        }

    except HTTPException:
        raise
    except (ValueError, TypeError, OverflowError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid customer data in CSV: {str(exc)}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process CSV: {str(exc)}",
        ) from exc


class RetentionRequest(BaseModel):
    customer: dict[str, Any] = Field(
        ...,
        max_length=100
    )
    churn_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0
    )


@app.post("/retention-recommendations")
def retention_recommendations(request: RetentionRequest):
    return generate_retention_plan(
        customer=request.customer,
        churn_probability=request.churn_probability,
    )