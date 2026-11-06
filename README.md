# Credit Approval Classification — Model Comparison & Streamlit App

## a. Problem statement
Predict whether a credit card application is **approved (+)** or **rejected (-)** from 15 anonymised applicant attributes. This is a binary classification task. Five classifiers (Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM) are trained on the same data, compared on six metrics, and exposed through a Streamlit app that scores an uploaded test CSV with the saved models (no retraining).

## b. Dataset description
- **Source:** UCI Machine Learning Repository — *Credit Approval* (https://archive.ics.uci.edu/dataset/27/credit+approval)
- **Instances:** 690 (307 approved `+`, 383 rejected `-`; 44.5% / 55.5%)
- **Features:** 15 (A1–A15) + target A16. Names and values were anonymised by the donor, so attributes have no stated real-world meaning.
  - 6 numeric: A2, A3, A8, A11, A14, A15
  - 9 categorical: A1, A4, A5, A6, A7, A9, A10, A12, A13
- **Missing values:** marked `?` in the raw file; 37 rows (5%) have at least one (A1, A2, A4, A5, A6, A7, A14).
- **Preprocessing (inside each saved pipeline):** numeric → median imputation → `log1p` → standardisation; categorical → most-frequent imputation → one-hot (unseen categories ignored). Because preprocessing is part of each saved model, the app can accept raw rows with blanks or `?`.
- **Split:** stratified 80/20, `random_state=11` → 552 train / 138 test rows. Hyperparameters were chosen by 5-fold stratified CV (scoring = ROC-AUC) on the training portion only; the test set was used once for the final numbers.

## c. GitHub repository link
https://github.com/debabghosh21/Credit_Approval_Classification

Live app: http://localhost:8501/ or https://creditapprovalclassification-fr8wnmpsmtmeezb3pxn2af.streamlit.app/

## d. Models used

### Comparison table (hold-out test set, 138 rows)
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8551 | 0.9304 | 0.8154 | 0.8689 | 0.8413 | 0.7094 |
| Decision Tree | 0.8261 | 0.9093 | 0.7342 | 0.9508 | 0.8286 | 0.6807 |
| KNN | 0.8116 | 0.8843 | 0.8302 | 0.7213 | 0.7719 | 0.6172 |
| Naive Bayes | 0.8623 | 0.9240 | 0.8182 | 0.8852 | 0.8504 | 0.7252 |
| Support Vector Machine | 0.8623 | 0.9174 | 0.8281 | 0.8689 | 0.8480 | 0.7230 |

Selected hyperparameters: Logistic Regression C=1; Decision Tree max_depth=3, min_samples_leaf=10; KNN k=15 (distance-weighted); Naive Bayes (Gaussian) var_smoothing=0.1; SVM RBF kernel, C=1, gamma=scale.

### Observations on model performance
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest AUC (0.930) and a well-balanced precision/recall (0.815 / 0.869). The data is largely separable by a linear boundary once numeric features are log-scaled and categoricals are one-hot encoded, so a simple, interpretable model is already very competitive. |
| Decision Tree | Shallow tree (depth 3) gave the highest recall (0.951; only 3 approvals missed) but the lowest precision (0.734; 21 rejected applicants predicted approved). It is the most "lenient" model, which would be risky if wrongly approving credit is costly. The depth limit was needed to avoid overfitting. |
| KNN | Weakest overall (accuracy 0.812, AUC 0.884, MCC 0.617). Precision is fine (0.830) but recall is low (0.721; 17 approvals missed). One-hot encoding adds many binary dimensions that dilute Euclidean distance, which hurts a distance-based method. |
| Naive Bayes | Best accuracy, F1 and MCC on the test set (0.862 / 0.850 / 0.725) despite its independence assumption. Strong smoothing (var_smoothing=0.1) helped, and the model has low variance, which suits a small training set of 552 rows. |
| Support Vector Machine | Ties Naive Bayes on accuracy (0.862) with MCC 0.723 and the best cross-validated AUC on the training data (0.931). Its test AUC (0.917) is slightly below Logistic Regression, suggesting the RBF kernel adds little beyond a linear boundary here. |

**Overall winner for this dataset:** Naive Bayes has the best accuracy, F1 and MCC on the test set, but the margin is very small. With only 138 test rows, one extra correct prediction changes accuracy by about 0.7 percentage points, and Logistic Regression and SVM are within noise of it (Logistic Regression has the best test AUC; SVM the best CV AUC). Naive Bayes, SVM and Logistic Regression are effectively tied; KNN is clearly the weakest. If a single model must be chosen I would pick **Naive Bayes** on the reported metrics, with Logistic Regression as the safer, more interpretable alternative.

## Repository structure
```
app.py               Streamlit app
scoring.py           upload validation + evaluation helpers
metrics_utils.py     metric computation shared by training and app
train_models.py      trains/tunes the 5 models, saves them, writes test_data.csv
requirements.txt
test_data.csv        138 held-out rows (features + target) for the app
data/                raw crx.data and crx.names from UCI
model/               saved pipelines (*.joblib), metrics.csv, best_params.json
```

## Run locally
```
pip install -r requirements.txt
python train_models.py      # optional: regenerates model/ and test_data.csv
streamlit run app.py
```

## Streamlit app features
- CSV upload (sample `test_data.csv` downloadable from the sidebar)
- Model selection dropdown (5 models)
- Evaluation metrics: Accuracy, AUC, Precision, Recall, F1, MCC
- Confusion matrix and classification report
- Side-by-side table of all five models on the uploaded data
- Saved models are loaded from `model/`; nothing is retrained on the upload
