
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import classification_report

# Project folders
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "churn_synthetic.csv"
MODEL_PATH = MODEL_DIR / "churn_model.json"

# Generate reproducible synthetic customer data
rng = np.random.default_rng(42)
n = 5000

tenure = rng.integers(0, 73, n)
monthly_charges = rng.uniform(20, 120, n).round(2)
contract = rng.choice(
    ["Monthly", "One year", "Two year"],
    size=n,
    p=[0.55, 0.25, 0.20],
)
internet_fiber = rng.integers(0, 2, n)
tech_support = rng.integers(0, 2, n)
paperless_billing = rng.integers(0, 2, n)
senior_citizen = rng.integers(0, 2, n)
dependents = rng.integers(0, 2, n)

# Convert input features to numeric columns for XGBoost
df = pd.DataFrame({
    "tenure": tenure,
    "monthly_charges": monthly_charges,
    "contract_monthly": (contract == "Monthly").astype(int),
    "contract_one_year": (contract == "One year").astype(int),
    "contract_two_year": (contract == "Two year").astype(int),
    "internet_fiber": internet_fiber,
    "tech_support": tech_support,
    "paperless_billing": paperless_billing,
    "senior_citizen": senior_citizen,
    "dependents": dependents,
})

# Create synthetic churn labels with realistic directional patterns.
# These labels are simulated, not real customer outcomes.
log_odds = (
    1.0
    - 0.045 * tenure
    + 0.025 * (monthly_charges - 65)
    + 1.15 * df["contract_monthly"]
    - 0.45 * df["contract_one_year"]
    - 1.00 * df["contract_two_year"]
    + 0.55 * internet_fiber
    - 0.85 * tech_support
    + 0.30 * paperless_billing
    + 0.25 * senior_citizen
    - 0.35 * dependents
)

probability = 1 / (1 + np.exp(-log_odds))
df["churn"] = rng.binomial(1, probability)

# Save the generated dataset
df.to_csv(DATA_PATH, index=False)

# Prepare features and target
feature_columns = [
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

X = df[feature_columns]
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Train XGBoost classifier
model = XGBClassifier(
    n_estimators=250,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=2,
)

model.fit(X_train, y_train)

# Evaluate on held-out synthetic test data
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\n--- XGBoost Model Evaluation ---")
print(f"Training records: {len(X_train)}")
print(f"Test records: {len(X_test)}")
print(f"Test accuracy: {accuracy_score(y_test, predictions):.3f}")
print(f"Test ROC-AUC: {roc_auc_score(y_test, probabilities):.3f}")
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

# Save model in XGBoost JSON format
model.save_model(str(MODEL_PATH))

print("\nTraining completed successfully!")
print(f"Dataset saved: {DATA_PATH}")
print(f"Model saved: {MODEL_PATH}")
print(f"Features: {feature_columns}")