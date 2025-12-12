import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
import os

from utils import load_and_clean_data

CSV_PATH = "predictive_maintenance.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
X, y = load_and_clean_data(CSV_PATH)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=-1
    )
}

best_model = None
best_score = 0

# Train and evaluate each model
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    score = accuracy_score(y_test, y_pred)

    print(f"{model_name} Accuracy: {score:.4f}")

    # Save model
    model_path = os.path.join(MODEL_DIR, f"{model_name}_pipeline.pkl")
    joblib.dump(pipeline, model_path)
    print(f"Saved: {model_path}")

    if score > best_score:
        best_score = score
        best_model = pipeline
        best_name = model_name

# Save best model
joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
print(f"\nBest Model: {best_name} with accuracy {best_score:.4f}")
