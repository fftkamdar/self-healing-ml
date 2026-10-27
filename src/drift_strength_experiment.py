
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
# 1. PATHS AND CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
MODEL_PATH = ROOT / "models" / "baseline_churn_pipeline.joblib"
REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

PSI_THRESHOLD = 0.25
RANDOM_STATE = 42


# =========================================================
# 2. LOAD DATA AND MODEL
# =========================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Baseline model not found: {MODEL_PATH}")

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

# Keep an adaptation subset separate from final evaluation.
X_adapt, X_eval, y_adapt, y_eval = train_test_split(
    X_test,
    y_test,
    test_size=0.5,
    random_state=21,
    stratify=y_test,
)

model = joblib.load(MODEL_PATH)


# =========================================================
# 3. PSI FUNCTIONS
# =========================================================

def numeric_psi(reference, current, bins=10):
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


def get_max_psi(reference_df, current_df):
    scores = []

    for column in reference_df.columns:
        reference = reference_df[column]
        current = current_df[column]

        if (
            pd.api.types.is_numeric_dtype(reference)
            and not pd.api.types.is_bool_dtype(reference)
        ):
            score = numeric_psi(reference, current)
        else:
            score = categorical_psi(reference, current)

        if not pd.isna(score):
            scores.append(score)

    return float(max(scores)) if scores else 0.0


# =========================================================
# 4. DEFINE DRIFT LEVELS
# =========================================================

DRIFT_LEVELS = [
    {
        "name": "No shift",
        "tenure_multiplier": 1.0,
        "charges_multiplier": 1.0,
    },
    {
        "name": "Mild shift",
        "tenure_multiplier": 1.1,
        "charges_multiplier": 1.05,
    },
    {
        "name": "Moderate shift",
        "tenure_multiplier": 1.3,
        "charges_multiplier": 1.1,
    },
    {
        "name": "Stronger shift",
        "tenure_multiplier": 1.5,
        "charges_multiplier": 1.2,
    },
]


def apply_drift(data, tenure_multiplier, charges_multiplier):
    shifted = data.copy()

    shifted["tenure"] = (
        shifted["tenure"] * tenure_multiplier
    ).clip(upper=72)

    shifted["MonthlyCharges"] = (
        shifted["MonthlyCharges"] * charges_multiplier
    ).clip(upper=120)

    return shifted


# =========================================================
# 5. EVALUATE EACH CONDITION
# =========================================================

results = []

for level in DRIFT_LEVELS:
    X_current = apply_drift(
        X_eval,
        level["tenure_multiplier"],
        level["charges_multiplier"],
    )

    max_psi = get_max_psi(X_train, X_current)

    if max_psi >= PSI_THRESHOLD:
        decision = "RETRAINING_CONSIDERED"
    else:
        decision = "SKIP_RETRAINING"

    predictions = model.predict(X_current)
    probabilities = model.predict_proba(X_current)[:, 1]

    results.append({
        "drift_level": level["name"],
        "tenure_multiplier": level["tenure_multiplier"],
        "monthly_charges_multiplier": level["charges_multiplier"],
        "max_psi": max_psi,
        "decision": decision,
        "evaluation_rows": int(len(y_eval)),
        "accuracy": accuracy_score(y_eval, predictions),
        "precision": precision_score(
            y_eval, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_eval, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_eval, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_eval, probabilities),
    })


# =========================================================
# 6. SAVE RESULTS
# =========================================================

results_df = pd.DataFrame(results)

output_path = REPORT_DIR / "drift_strength_comparison.csv"
results_df.to_csv(output_path, index=False)

print("\nDRIFT STRENGTH EXPERIMENT")
print("=" * 110)

print(
    results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

print("\nSaved report:")
print(output_path)

print("\nExperiment complete.")
print(
    "All drift conditions are synthetic. Original evaluation "
    "labels are retained after feature transformations."
)