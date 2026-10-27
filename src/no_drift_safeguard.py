
from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split


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
# 2. LOAD DATA
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

# Split test data into adaptation and evaluation subsets.
X_adapt, X_eval, y_adapt, y_eval = train_test_split(
    X_test,
    y_test,
    test_size=0.5,
    random_state=21,
    stratify=y_test,
)

baseline_model = joblib.load(MODEL_PATH)


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


def diagnose_drift(reference_df, current_df):
    records = []

    for column in reference_df.columns:
        ref = reference_df[column]
        cur = current_df[column]

        if (
            pd.api.types.is_numeric_dtype(ref)
            and not pd.api.types.is_bool_dtype(ref)
        ):
            score = numeric_psi(ref, cur)
        else:
            score = categorical_psi(ref, cur)

        if pd.isna(score):
            level = "Not calculated"
        elif score >= HIGH_PSI_THRESHOLD:
            level = "High"
        elif score >= 0.10:
            level = "Moderate"
        else:
            level = "Low"

        records.append({
            "feature": column,
            "psi": score,
            "drift_level": level,
        })

    return pd.DataFrame(records)


# =========================================================
# 4. CREATE TWO SCENARIOS
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


# Scenario 1: unchanged data.
X_no_drift = X_adapt.copy()

# Scenario 2: controlled synthetic drift.
X_with_drift = apply_drift(X_adapt)


# =========================================================
# 5. DECISION RULE
# =========================================================

def decide_action(reference, current, scenario_name):
    diagnosis = diagnose_drift(reference, current)

    valid_scores = diagnosis["psi"].dropna()

    max_psi = (
        float(valid_scores.max())
        if not valid_scores.empty
        else 0.0
    )

    high_features = diagnosis.loc[
        diagnosis["drift_level"] == "High",
        "feature",
    ].tolist()

    if max_psi >= HIGH_PSI_THRESHOLD:
        action = "RETRAINING_CONSIDERED"
    else:
        action = "SKIP_RETRAINING"

    return {
        "scenario": scenario_name,
        "max_psi": max_psi,
        "high_drift_features": high_features,
        "decision": action,
        "diagnosis": diagnosis,
    }


no_drift_result = decide_action(
    X_train,
    X_no_drift,
    "No-drift control",
)

drift_result = decide_action(
    X_train,
    X_with_drift,
    "Synthetic drift",
)


# =========================================================
# 6. SAVE REPORTS
# =========================================================

summary_rows = []

for result in [no_drift_result, drift_result]:
    summary_rows.append({
        "scenario": result["scenario"],
        "max_psi": result["max_psi"],
        "high_drift_features": ", ".join(
            result["high_drift_features"]
        ),
        "decision": result["decision"],
    })

summary_df = pd.DataFrame(summary_rows)

summary_csv = REPORT_DIR / "no_drift_safeguard_results.csv"
summary_df.to_csv(summary_csv, index=False)

no_drift_result["diagnosis"].to_csv(
    REPORT_DIR / "no_drift_feature_diagnosis.csv",
    index=False,
)

drift_result["diagnosis"].to_csv(
    REPORT_DIR / "synthetic_drift_feature_diagnosis.csv",
    index=False,
)

json_summary = {
    "experiment": "No-drift safeguard",
    "threshold": HIGH_PSI_THRESHOLD,
    "scenarios": summary_rows,
    "interpretation": (
        "The decision rule skips retraining below the PSI threshold "
        "and allows retraining to be considered at or above it. "
        "This is a simulated decision policy; it does not prove "
        "that retraining improves predictive performance."
    ),
}

json_path = REPORT_DIR / "no_drift_safeguard_summary.json"

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(json_summary, file, indent=2)

print("\nNO-DRIFT SAFEGUARD RESULTS")
print("=" * 72)
print(summary_df.to_string(index=False))

print("\nSaved reports:")
print(summary_csv)
print(REPORT_DIR / "no_drift_feature_diagnosis.csv")
print(REPORT_DIR / "synthetic_drift_feature_diagnosis.csv")
print(json_path)

print("\nSafeguard experiment complete.")