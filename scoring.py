"""Upload validation + evaluation logic for the app (no Streamlit imports, so it is unit-testable)."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from metrics_utils import compute_metrics

MODEL_DIR = Path(__file__).parent / "model"
FEATURES = [f"A{i}" for i in range(1, 16)]
TARGET = "A16"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


def load_models():
    return {name: joblib.load(MODEL_DIR / f) for name, f in MODEL_FILES.items()}


def prepare_upload(df):
    """Validate an uploaded frame and return (X, y). Raises ValueError with a readable message."""
    df = df.replace("?", np.nan)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError("Uploaded CSV is missing required columns: " + ", ".join(missing))
    labels = df[TARGET].astype(str).str.strip()
    bad = sorted(set(labels) - {"+", "-"})
    if bad:
        raise ValueError(f"Target column {TARGET} must contain only '+' or '-' (found: {bad[:5]})")
    num_cols = ["A2", "A3", "A8", "A11", "A14", "A15"]
    X = df[FEATURES].copy()
    X[num_cols] = X[num_cols].apply(pd.to_numeric, errors="coerce")
    return X, (labels == "+").astype(int)


def evaluate(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    metrics = compute_metrics(y, pred, proba) if y.nunique() == 2 else None
    cm = confusion_matrix(y, pred, labels=[0, 1])
    report = classification_report(
        y, pred, labels=[0, 1], target_names=["Rejected (-)", "Approved (+)"],
        output_dict=True, zero_division=0,
    )
    return metrics, cm, pd.DataFrame(report).T.round(3), pred, proba
