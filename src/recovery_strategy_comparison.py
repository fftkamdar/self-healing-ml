
"""
Recovery Strategy Comparison
----------------------------

Compares several model recovery strategies under synthetic feature drift.

Strategies:
1. Baseline: Keep the original model.
2. Full retraining: Retrain using the adaptation data.
3. Selective retraining: Retrain after excluding high-drift features.
4. Performance-aware selection: Select the candidate only if it passes
   the configured promotion gate.

This is a controlled experiment on a public dataset. It does not represent
a live production deployment.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


# ---------------------------------------------------------
# PATHS AND CONFIGURATION
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

PSI_THRESHOLD = 0.25
TENURE_MULTIPLIER = 1.5
MONTHLY_CHARGES_MULTIPLIER = 1.2

TARGET = "Churn"
DROP_COLUMNS = ["customerID"]

HIGH_DRIFT_FEATURES = ["tenure", "MonthlyCharges"]

OUTPUT_CSV = REPORTS_DIR / "recovery_strategy_comparison.csv"
OUTPUT_TXT = REPORTS_DIR / "recovery_strategy_comparison_report.txt"


# ---------------------------------------------------------
# DATA PREPARATION
# ---------------------------------------------------------

def load_data():
    """Load the Telco dataset and prepare its target and features."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Make sure Telco-Customer-Churn.csv is inside data/raw/."
        )

    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' was not found.")

    # Convert TotalCharges to numeric. Blank values become NaN and are
    # handled later by the numeric imputer.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"], errors="coerce"
        )

    y = df[TARGET].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError(
            "Unexpected values were found in the Churn target column."
        )

    X = df.drop(columns=[TARGET] + DROP_COLUMNS, errors="ignore")

    return X, y.astype(int)


def apply_synthetic_drift(X):
    """
    Apply the project's documented synthetic shifts.

    This changes tenure and MonthlyCharges to simulate a changed
    incoming data distribution. It is not real-world drift evidence.
    """

    shifted = X.copy()

    if "tenure" in shifted.columns:
        shifted["tenure"] = (
            pd.to_numeric(shifted["tenure"], errors="coerce")
            * TENURE_MULTIPLIER
        ).clip(upper=72)

    if "MonthlyCharges" in shifted.columns:
        shifted["MonthlyCharges"] = (
            pd.to_numeric(shifted["MonthlyCharges"], errors="coerce")
            * MONTHLY_CHARGES_MULTIPLIER
        ).clip(upper=120)

    return shifted


# ---------------------------------------------------------
# MODEL CREATION
# ---------------------------------------------------------

def build_model(X):
    """Create a preprocessing and logistic-regression pipeline."""

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [
        column for column in X.columns
        if column not in numeric_features
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
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

    return model


def evaluate_model(model, X_eval, y_eval):
    """Calculate common binary classification metrics."""

    predictions = model.predict(X_eval)
    probabilities = model.predict_proba(X_eval)[:, 1]

    return {
        "accuracy": accuracy_score(y_eval, predictions),
        "precision": precision_score(
            y_eval, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_eval, predictions, zero_division=0
        ),
        "f1": f1_score(y_eval, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_eval, probabilities),
    }


# ---------------------------------------------------------
# PROMOTION GATE
# ---------------------------------------------------------

def passes_promotion_gate(baseline_metrics, candidate_metrics):
    """
    Require the candidate to be no worse than baseline on all
    configured promotion metrics.

    This mirrors the strict comparison used in the project:
    recall, F1-score, and ROC-AUC.
    """

    required_metrics = ["recall", "f1", "roc_auc"]

    return all(
        candidate_metrics[metric] >= baseline_metrics[metric]
        for metric in required_metrics
    )


# ---------------------------------------------------------
# EXPERIMENT
# ---------------------------------------------------------

def main():
    print("=" * 72)
    print("RECOVERY STRATEGY COMPARISON")
    print("=" * 72)

    X, y = load_data()

    # First split: 80% initial training, 20% reserved for adaptation
    # and final evaluation.
    X_train, X_remaining, y_train, y_remaining = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Divide the remaining 20% equally.
    X_adapt, X_eval, y_adapt, y_eval = train_test_split(
        X_remaining,
        y_remaining,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_remaining,
    )

    # Simulate distribution change in both adaptation and evaluation
    # features. Labels remain unchanged.
    X_adapt_shifted = apply_synthetic_drift(X_adapt)
    X_eval_shifted = apply_synthetic_drift(X_eval)

    print(f"\nDataset rows: {len(X)}")
    print(f"Initial training rows: {len(X_train)}")
    print(f"Adaptation rows: {len(X_adapt_shifted)}")
    print(f"Evaluation rows: {len(X_eval_shifted)}")
    print("\nSynthetic drift applied to:")
    print(f"  tenure multiplier: {TENURE_MULTIPLIER}")
    print(f"  MonthlyCharges multiplier: {MONTHLY_CHARGES_MULTIPLIER}")

    results = []

    # -----------------------------------------------------
    # STRATEGY 1: KEEP BASELINE
    # -----------------------------------------------------

    baseline_model = build_model(X_train)
    baseline_model.fit(X_train, y_train)

    baseline_metrics = evaluate_model(
        baseline_model, X_eval_shifted, y_eval
    )

    results.append(
        {
            "strategy": "Baseline - Keep Existing Model",
            "model_selected": "Baseline",
            **baseline_metrics,
        }
    )

    # -----------------------------------------------------
    # STRATEGY 2: FULL RETRAINING
    # -----------------------------------------------------

    X_full_train = pd.concat(
        [X_train, X_adapt_shifted], ignore_index=True
    )
    y_full_train = pd.concat(
        [y_train, y_adapt], ignore_index=True
    )

    full_model = build_model(X_full_train)
    full_model.fit(X_full_train, y_full_train)

    full_metrics = evaluate_model(
        full_model, X_eval_shifted, y_eval
    )

    results.append(
        {
            "strategy": "Full Retraining",
            "model_selected": "Retrained Candidate",
            **full_metrics,
        }
    )

    # -----------------------------------------------------
    # STRATEGY 3: SELECTIVE FEATURE RETRAINING
    # -----------------------------------------------------

    # Exclude the features that were intentionally shifted in this
    # experiment. This is a comparison strategy, not a claim that
    # dropping drifted features is always appropriate.
    excluded = [
        feature for feature in HIGH_DRIFT_FEATURES
        if feature in X_train.columns
    ]

    selected_features = [
        feature for feature in X_train.columns
        if feature not in excluded
    ]

    if not selected_features:
        raise ValueError(
            "No features remain after selective feature exclusion."
        )

    X_selective_train = X_full_train[selected_features]
    X_selective_eval = X_eval_shifted[selected_features]

    selective_model = build_model(X_selective_train)
    selective_model.fit(X_selective_train, y_full_train)

    selective_metrics = evaluate_model(
        selective_model, X_selective_eval, y_eval
    )

    results.append(
        {
            "strategy": "Selective Retraining - Exclude Shifted Features",
            "model_selected": "Selective Candidate",
            **selective_metrics,
        }
    )

    # -----------------------------------------------------
    # STRATEGY 4: PERFORMANCE-AWARE SELECTION
    # -----------------------------------------------------

    gate_passed = passes_promotion_gate(
        baseline_metrics, full_metrics
    )

    if gate_passed:
        chosen_name = "Retrained Candidate"
        chosen_metrics = full_metrics
    else:
        chosen_name = "Baseline"
        chosen_metrics = baseline_metrics

    results.append(
        {
            "strategy": "Performance-Aware Selection",
            "model_selected": chosen_name,
            **chosen_metrics,
        }
    )

    # -----------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------

    results_df = pd.DataFrame(results)

    metric_columns = [
        "accuracy", "precision", "recall", "f1", "roc_auc"
    ]
    results_df[metric_columns] = results_df[metric_columns].round(4)

    results_df.to_csv(OUTPUT_CSV, index=False)

    lines = [
        "RECOVERY STRATEGY COMPARISON REPORT",
        "=" * 72,
        "",
        f"Dataset rows: {len(X)}",
        f"Initial training rows: {len(X_train)}",
        f"Adaptation rows: {len(X_adapt_shifted)}",
        f"Evaluation rows: {len(X_eval_shifted)}",
        "",
        "Synthetic drift configuration:",
        f"  tenure multiplier: {TENURE_MULTIPLIER}",
        f"  MonthlyCharges multiplier: {MONTHLY_CHARGES_MULTIPLIER}",
        "",
        f"Features excluded by selective strategy: {excluded}",
        "",
        "Promotion gate metrics: recall, f1, roc_auc",
        f"Full retraining candidate passed gate: {gate_passed}",
        f"Performance-aware selection chose: {chosen_name}",
        "",
        "Results:",
        results_df.to_string(index=False),
        "",
        "Interpretation notes:",
        "- The shifts are synthetic and do not establish real-world drift.",
        "- The same held-out evaluation rows are used for each strategy.",
        "- Retraining is not assumed to improve every metric.",
        "- Selective feature exclusion may remove useful predictive signal.",
        "- This is an offline experiment, not a live production deployment.",
        "",
    ]

    report_text = "\n".join(lines)
    OUTPUT_TXT.write_text(report_text, encoding="utf-8")

    print("\n" + results_df.to_string(index=False))

    print("\nPromotion gate:")
    print(f"  Full retraining candidate passed: {gate_passed}")
    print(f"  Selected model: {chosen_name}")

    print("\nSaved outputs:")
    print(OUTPUT_CSV)
    print(OUTPUT_TXT)

    print("\nExperiment completed successfully.")


if __name__ == "__main__":
    main()
    