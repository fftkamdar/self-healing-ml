
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ---------------------------------------------------------
# 1. PROJECT PATHS AND EXPERIMENT SETTINGS
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
MODEL_PATH = ROOT / "models" / "baseline_churn_pipeline.joblib"
REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Adaptation data is kept separate from final evaluation data.
ADAPTATION_SIZE = 0.5

# Controlled synthetic feature drift, matching the earlier project.
TENURE_MULTIPLIER = 1.5
MONTHLY_CHARGES_MULTIPLIER = 1.2


# ---------------------------------------------------------
# 2. LOAD AND PREPARE DATA
# ---------------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Baseline model not found: {MODEL_PATH}")

df = pd.read_csv(DATA_PATH)

# Convert TotalCharges to numeric.
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Remove the customer identifier from model inputs.
df = df.drop(columns=["customerID"], errors="ignore")

# Encode target labels.
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

if df["Churn"].isna().any():
    raise ValueError("Unexpected or missing values found in Churn.")

X = df.drop(columns=["Churn"])
y = df["Churn"].astype(int)

# Use the same deterministic stratified split convention
# as the existing baseline experiment.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y,
)

# Divide the held-out test set into adaptation and evaluation
# subsets. The final evaluation rows are not used for fitting.
X_adapt, X_eval, y_adapt, y_eval = train_test_split(
    X_test,
    y_test,
    test_size=ADAPTATION_SIZE,
    random_state=21,
    stratify=y_test,
)

# Load the existing fitted baseline pipeline.
baseline_model = joblib.load(MODEL_PATH)


# ---------------------------------------------------------
# 3. SYNTHETIC DRIFT TRANSFORMATION
# ---------------------------------------------------------

def apply_synthetic_drift(data):
    """Return a copy with controlled shifts to two features."""
    shifted = data.copy()

    shifted["tenure"] = (
        shifted["tenure"] * TENURE_MULTIPLIER
    ).clip(upper=72)

    shifted["MonthlyCharges"] = (
        shifted["MonthlyCharges"] * MONTHLY_CHARGES_MULTIPLIER
    ).clip(upper=120)

    return shifted


X_adapt_drift = apply_synthetic_drift(X_adapt)
X_eval_drift = apply_synthetic_drift(X_eval)


# ---------------------------------------------------------
# 4. BUILD A FRESH PIPELINE FOR RETRAINING
# ---------------------------------------------------------

numeric_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    exclude=["number"]
).columns.tolist()

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

def build_model():
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


# ---------------------------------------------------------
# 5. METRIC CALCULATION
# ---------------------------------------------------------

def evaluate_model(model, X_data, y_true, condition):
    predictions = model.predict(X_data)
    probabilities = model.predict_proba(X_data)[:, 1]

    return {
        "condition": condition,
        "rows_evaluated": int(len(y_true)),
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_true, probabilities),
    }


# ---------------------------------------------------------
# 6. RUN THE THREE EXPERIMENT CONDITIONS
# ---------------------------------------------------------

results = []

# A: Original model, unchanged evaluation data.
results.append(
    evaluate_model(
        baseline_model,
        X_eval,
        y_eval,
        "A - No drift, no retraining",
    )
)

# B: Original model, synthetically drifted evaluation data.
results.append(
    evaluate_model(
        baseline_model,
        X_eval_drift,
        y_eval,
        "B - Drift, no retraining",
    )
)

# C: Retrain using original training data plus the separate
# drifted adaptation subset. Evaluate on the same drifted
# evaluation rows as condition B.
X_retrain = pd.concat(
    [X_train, X_adapt_drift],
    ignore_index=True,
)

y_retrain = pd.concat(
    [y_train, y_adapt],
    ignore_index=True,
)

retrained_model = build_model()
retrained_model.fit(X_retrain, y_retrain)

results.append(
    evaluate_model(
        retrained_model,
        X_eval_drift,
        y_eval,
        "C - Drift, with retraining",
    )
)


# ---------------------------------------------------------
# 7. SAVE RESULTS
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

csv_path = REPORT_DIR / "controlled_experiment_results.csv"
json_path = REPORT_DIR / "controlled_experiment_summary.json"
model_output = ROOT / "models" / "controlled_retrained_pipeline.joblib"

results_df.to_csv(csv_path, index=False)

joblib.dump(retrained_model, model_output)

summary = {
    "experiment": "Controlled synthetic drift experiment",
    "dataset": DATA_PATH.name,
    "random_state": RANDOM_STATE,
    "adaptation_rows": int(len(y_adapt)),
    "evaluation_rows": int(len(y_eval)),
    "drift": {
        "tenure_multiplier": TENURE_MULTIPLIER,
        "tenure_upper_clip": 72,
        "monthly_charges_multiplier": MONTHLY_CHARGES_MULTIPLIER,
        "monthly_charges_upper_clip": 120,
    },
    "important_limitation": (
        "Feature drift is synthetically applied while original "
        "labels are retained. Results do not establish how a "
        "real production model would perform under real drift."
    ),
    "results": results_df.to_dict(orient="records"),
}

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2)

print("\nCONTROLLED EXPERIMENT RESULTS")
print("=" * 72)
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\nSaved files:")
print(csv_path)
print(json_path)
print(model_output)

print("\nNote: this is a controlled synthetic experiment,")
print("not a live production drift or recovery test.")