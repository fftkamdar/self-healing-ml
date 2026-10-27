
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# =========================================================
# 1. PATHS AND SETTINGS
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
MODEL_PATH = ROOT / "models" / "baseline_churn_pipeline.joblib"
REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

HIGH_PSI_THRESHOLD = 0.25
RANDOM_STATE = 42


# =========================================================
# 2. LOAD AND SPLIT DATA
# =========================================================

df = pd.read_csv(DATA_PATH)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

df = df.drop(columns=["customerID"], errors="ignore")
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

if df["Churn"].isna().any():
    raise ValueError("Unexpected or missing churn labels.")

X = df.drop(columns=["Churn"])
y = df["Churn"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y,
)

X_adapt, X_eval, y_adapt, y_eval = train_test_split(
    X_test,
    y_test,
    test_size=0.5,
    random_state=21,
    stratify=y_test,
)

model = joblib.load(MODEL_PATH)


# =========================================================
# 3. SYNTHETIC DRIFT
# =========================================================

def apply_drift(data):
    shifted = data.copy()

    shifted["tenure"] = (
        shifted["tenure"] * 1.5
    ).clip(upper=72)

    shifted["MonthlyCharges"] = (
        shifted["MonthlyCharges"] * 1.2
    ).clip(upper=120)

    return shifted


X_eval_drift = apply_drift(X_eval)


# =========================================================
# 4. PSI CALCULATION
# =========================================================

def calculate_psi(reference, current, bins=10):
    reference = pd.to_numeric(reference, errors="coerce")
    current = pd.to_numeric(current, errors="coerce")

    ref_missing = reference.isna()
    cur_missing = current.isna()

    ref = reference[~ref_missing]
    cur = current[~cur_missing]

    if len(ref) == 0 or len(cur) == 0:
        return np.nan

    edges = np.unique(
        np.quantile(ref, np.linspace(0, 1, bins + 1))
    )

    if len(edges) < 2:
        return np.nan

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts = pd.cut(
        ref, bins=edges, include_lowest=True
    ).value_counts(sort=False).to_numpy(dtype=float)

    cur_counts = pd.cut(
        cur, bins=edges, include_lowest=True
    ).value_counts(sort=False).to_numpy(dtype=float)

    ref_counts = np.append(ref_counts, ref_missing.sum())
    cur_counts = np.append(cur_counts, cur_missing.sum())

    epsilon = 1e-6

    ref_pct = (ref_counts + epsilon) / (
        ref_counts.sum() + epsilon * len(ref_counts)
    )
    cur_pct = (cur_counts + epsilon) / (
        cur_counts.sum() + epsilon * len(cur_counts)
    )

    return float(
        np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    )


def categorical_psi(reference, current):
    reference = reference.astype("string").fillna("__MISSING__")
    current = current.astype("string").fillna("__MISSING__")

    categories = sorted(
        set(reference.unique()) | set(current.unique())
    )

    ref_counts = reference.value_counts().reindex(
        categories, fill_value=0
    ).to_numpy(dtype=float)

    cur_counts = current.value_counts().reindex(
        categories, fill_value=0
    ).to_numpy(dtype=float)

    epsilon = 1e-6

    ref_pct = (ref_counts + epsilon) / (
        ref_counts.sum() + epsilon * len(ref_counts)
    )
    cur_pct = (cur_counts + epsilon) / (
        cur_counts.sum() + epsilon * len(cur_counts)
    )

    return float(
        np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    )


def maximum_psi(reference_df, current_df):
    scores = []

    for column in reference_df.columns:
        ref = reference_df[column]
        cur = current_df[column]

        if (
            pd.api.types.is_numeric_dtype(ref)
            and not pd.api.types.is_bool_dtype(ref)
        ):
            score = calculate_psi(ref, cur)
        else:
            score = categorical_psi(ref, cur)

        if not pd.isna(score):
            scores.append(score)

    return max(scores) if scores else 0.0


# =========================================================
# 5. EVALUATION
# =========================================================

def evaluate(features, labels, scenario):
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    max_psi = maximum_psi(X_train, features)

    decision = (
        "RETRAINING_CONSIDERED"
        if max_psi >= HIGH_PSI_THRESHOLD
        else "SKIP_RETRAINING"
    )

    return {
        "scenario": scenario,
        "rows": int(len(labels)),
        "max_psi": float(max_psi),
        "decision": decision,
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(
            labels, predictions, zero_division=0
        ),
        "recall": recall_score(
            labels, predictions, zero_division=0
        ),
        "f1": f1_score(
            labels, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(labels, probabilities),
    }


results = [
    evaluate(
        X_eval,
        y_eval,
        "No-drift control",
    ),
    evaluate(
        X_eval_drift,
        y_eval,
        "Synthetic drift",
    ),
]

results_df = pd.DataFrame(results)

output_path = REPORT_DIR / "drift_performance_comparison.csv"
results_df.to_csv(output_path, index=False)

print("\nDRIFT AND PERFORMANCE COMPARISON")
print("=" * 100)
print(
    results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

print("\nSaved:", output_path)
print("\nExperiment complete.")
print(
    "Reminder: synthetic feature drift with unchanged labels "
    "does not establish real-world model degradation."
)