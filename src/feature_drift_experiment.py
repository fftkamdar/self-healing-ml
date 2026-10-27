
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


# =========================================================
# 1. PATHS AND SETTINGS
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
BASELINE_PATH = ROOT / "models" / "baseline_churn_pipeline.joblib"

REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# PSI interpretation thresholds.
MODERATE_PSI = 0.10
HIGH_PSI = 0.25

# Synthetic drift settings, matching earlier experiments.
TENURE_MULTIPLIER = 1.5
MONTHLY_CHARGES_MULTIPLIER = 1.2


# =========================================================
# 2. LOAD AND PREPARE DATA
# =========================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

if not BASELINE_PATH.exists():
    raise FileNotFoundError(
        f"Baseline model not found: {BASELINE_PATH}"
    )

df = pd.read_csv(DATA_PATH)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

df = df.drop(columns=["customerID"], errors="ignore")
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

if df["Churn"].isna().any():
    raise ValueError("Unexpected or missing Churn labels.")

X = df.drop(columns=["Churn"])
y = df["Churn"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y,
)

# Split the held-out test portion into adaptation and evaluation.
# Evaluation data is not used to fit the models.
X_adapt, X_eval, y_adapt, y_eval = train_test_split(
    X_test,
    y_test,
    test_size=0.5,
    random_state=21,
    stratify=y_test,
)

baseline_model = joblib.load(BASELINE_PATH)


# =========================================================
# 3. APPLY CONTROLLED SYNTHETIC DRIFT
# =========================================================

def apply_synthetic_drift(data):
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


# =========================================================
# 4. PSI FUNCTIONS
# =========================================================

def calculate_numeric_psi(reference, current, bins=10):
    """
    Calculate PSI using quantile bins from reference data.
    Missing values are placed in their own bin.
    """
    ref = pd.to_numeric(reference, errors="coerce")
    cur = pd.to_numeric(current, errors="coerce")

    ref_missing = ref.isna()
    cur_missing = cur.isna()

    ref_valid = ref[~ref_missing]
    cur_valid = cur[~cur_missing]

    if len(ref_valid) == 0 or len(cur_valid) == 0:
        return np.nan

    # Quantile edges are learned from reference values only.
    edges = np.unique(
        np.quantile(
            ref_valid,
            np.linspace(0, 1, bins + 1)
        )
    )

    # A constant reference feature has no useful quantile bins.
    if len(edges) < 2:
        return np.nan

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts = pd.cut(
        ref_valid, bins=edges, include_lowest=True
    ).value_counts(sort=False).to_numpy(dtype=float)

    cur_counts = pd.cut(
        cur_valid, bins=edges, include_lowest=True
    ).value_counts(sort=False).to_numpy(dtype=float)

    # Add the missing-value bin.
    ref_counts = np.append(ref_counts, ref_missing.sum())
    cur_counts = np.append(cur_counts, cur_missing.sum())

    # Smoothing prevents division by zero.
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


def calculate_categorical_psi(reference, current):
    """
    Calculate PSI over the union of categories.
    Missing values are treated as a category.
    """
    ref = reference.astype("string").fillna("__MISSING__")
    cur = current.astype("string").fillna("__MISSING__")

    categories = sorted(set(ref.unique()) | set(cur.unique()))

    ref_counts = ref.value_counts().reindex(
        categories, fill_value=0
    ).to_numpy(dtype=float)

    cur_counts = cur.value_counts().reindex(
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


def calculate_feature_psi(reference_df, current_df):
    rows = []

    for column in reference_df.columns:
        ref = reference_df[column]
        cur = current_df[column]

        if (
            pd.api.types.is_numeric_dtype(ref)
            and not pd.api.types.is_bool_dtype(ref)
        ):
            score = calculate_numeric_psi(ref, cur)
            method = "numeric quantile bins"
        else:
            score = calculate_categorical_psi(ref, cur)
            method = "categorical proportions"

        if pd.isna(score):
            level = "Not calculated"
        elif score >= HIGH_PSI:
            level = "High"
        elif score >= MODERATE_PSI:
            level = "Moderate"
        else:
            level = "Low"

        rows.append({
            "feature": column,
            "psi": score,
            "drift_level": level,
            "method": method,
        })

    return pd.DataFrame(rows).sort_values(
        "psi", ascending=False, na_position="last"
    )


# Diagnose drift using training data as reference and the
# drifted adaptation subset as current data.
drift_df = calculate_feature_psi(
    X_train,
    X_adapt_drift,
)

drift_csv = REPORT_DIR / "feature_level_drift_diagnosis.csv"
drift_df.to_csv(drift_csv, index=False)

# High-PSI features are excluded in strategy C.
high_drift_features = drift_df.loc[
    drift_df["drift_level"] == "High", "feature"
].tolist()

print("\nFEATURE-LEVEL DRIFT DIAGNOSIS")
print("=" * 72)
print(drift_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nFeatures marked High drift:", high_drift_features)


# =========================================================
# 5. MODEL PIPELINE FACTORY
# =========================================================

def build_model(feature_columns):
    numeric_features = [
        col for col in feature_columns
        if pd.api.types.is_numeric_dtype(X_train[col])
        and not pd.api.types.is_bool_dtype(X_train[col])
    ]

    categorical_features = [
        col for col in feature_columns
        if col not in numeric_features
    ]

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


# =========================================================
# 6. METRIC EVALUATION
# =========================================================

def evaluate_model(model, features, labels, strategy):
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    return {
        "strategy": strategy,
        "evaluation_rows": int(len(labels)),
        "features_used": int(features.shape[1]),
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


# =========================================================
# 7. STRATEGY A: NO RETRAINING
# =========================================================

results = []

results.append(
    evaluate_model(
        baseline_model,
        X_eval_drift,
        y_eval,
        "A - Baseline, no retraining",
    )
)


# =========================================================
# 8. STRATEGY B: FULL RETRAINING
# =========================================================

X_full_retrain = pd.concat(
    [X_train, X_adapt_drift],
    ignore_index=True,
)

y_full_retrain = pd.concat(
    [y_train, y_adapt],
    ignore_index=True,
)

all_features = X_train.columns.tolist()

full_model = build_model(all_features)
full_model.fit(X_full_retrain[all_features], y_full_retrain)

results.append(
    evaluate_model(
        full_model,
        X_eval_drift[all_features],
        y_eval,
        "B - Full retraining, all features",
    )
)

joblib.dump(
    full_model,
    MODEL_DIR / "feature_experiment_full_retrained.joblib",
)


# =========================================================
# 9. STRATEGY C: EXCLUDE HIGH-DRIFT FEATURES
# =========================================================

selected_features = [
    feature for feature in all_features
    if feature not in high_drift_features
]

if len(selected_features) == 0:
    print(
        "\nWARNING: Every feature was marked High drift. "
        "Feature exclusion cannot be evaluated."
    )
else:
    X_selected_retrain = X_full_retrain[selected_features]
    X_selected_eval = X_eval_drift[selected_features]

    selected_model = build_model(selected_features)
    selected_model.fit(X_selected_retrain, y_full_retrain)

    results.append(
        evaluate_model(
            selected_model,
            X_selected_eval,
            y_eval,
            "C - Drift-aware exclusion",
        )
    )

    joblib.dump(
        selected_model,
        MODEL_DIR / "feature_experiment_drift_aware.joblib",
    )


# =========================================================
# 10. SAVE AND DISPLAY RESULTS
# =========================================================

results_df = pd.DataFrame(results)

results_csv = REPORT_DIR / "feature_strategy_comparison.csv"
results_df.to_csv(results_csv, index=False)

summary = {
    "experiment": "Feature-level drift diagnosis and strategy comparison",
    "drift_type": "Controlled synthetic feature drift",
    "reference_rows": int(len(X_train)),
    "adaptation_rows": int(len(X_adapt_drift)),
    "evaluation_rows": int(len(X_eval_drift)),
    "psi_thresholds": {
        "moderate": MODERATE_PSI,
        "high": HIGH_PSI,
    },
    "high_drift_features": high_drift_features,
    "selected_features_after_exclusion": selected_features,
    "results": results_df.to_dict(orient="records"),
    "limitations": [
        "Feature drift is synthetically introduced.",
        "Original labels are retained after feature transformation.",
        "PSI measures distribution change, not predictive performance.",
        "Excluding drifted features may remove useful information.",
        "The experiment does not demonstrate live production recovery.",
    ],
}

summary_path = REPORT_DIR / "feature_strategy_summary.json"

with open(summary_path, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2)

print("\nSTRATEGY COMPARISON")
print("=" * 90)
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\nSaved reports:")
print(drift_csv)
print(results_csv)
print(summary_path)

print("\nExperiment complete.")