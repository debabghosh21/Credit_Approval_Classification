"""Train five classifiers on the UCI Credit Approval dataset and save them.

Run:  python train_models.py
Outputs: model/*.joblib, test_data.csv, model/metrics.csv, model/best_params.json
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import joblib

from metrics_utils import compute_metrics

warnings.filterwarnings("ignore")

SEED = 11
ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

COLS = [f"A{i}" for i in range(1, 17)]
NUM_COLS = ["A2", "A3", "A8", "A11", "A14", "A15"]
CAT_COLS = ["A1", "A4", "A5", "A6", "A7", "A9", "A10", "A12", "A13"]
TARGET = "A16"


def build_preprocessor():
    # numeric: median impute -> log1p (many columns are heavily right-skewed) -> standardise
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    # categorical: most-frequent impute -> one-hot (unseen categories ignored at inference)
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", num_pipe, NUM_COLS), ("cat", cat_pipe, CAT_COLS)])


MODELS = {
    "Logistic Regression": (
        LogisticRegression(max_iter=2000, random_state=SEED),
        {"clf__C": [0.01, 0.1, 1, 10, 100]},
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=SEED),
        {"clf__max_depth": [3, 4, 5, 6, 8, None], "clf__min_samples_leaf": [1, 5, 10, 20]},
    ),
    "KNN": (
        KNeighborsClassifier(),
        {"clf__n_neighbors": [3, 5, 7, 9, 11, 15, 21], "clf__weights": ["uniform", "distance"]},
    ),
    "Naive Bayes": (
        GaussianNB(),
        {"clf__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3, 1e-2, 1e-1]},
    ),
    "Support Vector Machine": (
        SVC(probability=True, random_state=SEED),
        {"clf__C": [0.1, 1, 10], "clf__kernel": ["rbf", "linear"], "clf__gamma": ["scale", 0.01]},
    ),
}


def load_raw(path):
    return pd.read_csv(path, header=None, names=COLS, na_values="?")


def main():
    df = load_raw(ROOT / "data" / "crx.data")
    X, y = df[COLS[:-1]], (df[TARGET] == "+").astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # test_data.csv keeps the ORIGINAL labels (+/-) and blanks for missing values
    test_out = X_te.copy()
    test_out[TARGET] = np.where(y_te == 1, "+", "-")
    test_out.to_csv(ROOT / "test_data.csv", index=False)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, best_params = [], {}
    for name, (clf, grid) in MODELS.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1).fit(X_tr, y_tr)
        best = gs.best_estimator_
        best_params[name] = {k: (v if v is not None else "None") for k, v in gs.best_params_.items()}
        proba = best.predict_proba(X_te)[:, 1]
        m = compute_metrics(y_te, best.predict(X_te), proba)
        m["cv_auc_train"] = round(gs.best_score_, 4)
        rows.append({"ML Model Name": name, **m})
        joblib.dump(best, MODEL_DIR / (name.lower().replace(" ", "_") + ".joblib"))
        print(f"{name:24s} {m}  best={gs.best_params_}")

    pd.DataFrame(rows).to_csv(MODEL_DIR / "metrics.csv", index=False)
    (MODEL_DIR / "best_params.json").write_text(json.dumps(best_params, indent=2))
    print("train rows:", len(X_tr), " test rows:", len(X_te))


if __name__ == "__main__":
    main()
