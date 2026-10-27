
from pathlib import Path

import numpy as np
import pandas as pd

from joblib import load
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
MODEL_PATH = ROOT / "models" / "baseline_churn_pipeline.joblib"
REPORT_DIR = ROOT / "reports"

RANDOM_STATE = 42


def apply_synthetic_drift(frame):
    """Apply the stronger synthetic shift used in earlier experiments."""
    shifted = frame.copy()

    shifted["tenure"] = (
        pd.to_numeric(shifted["tenure"], errors="coerce") * 1.5
    ).clip(upper=72)

    shifted["MonthlyCharges"] = (
        pd.to_numeric(shifted["MonthlyCharges"], errors="coerce") * 1.2
    ).clip(upper=120)

    return shifted


def calculate_metrics(model, X, y):
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    return {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(
            y, predictions, zero_division=0
        ),
        "recall": recall_score(
            y, predictions, zero_division=0
        ),
        "f1": f1_score(y, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilities),
    }


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Baseline model not found: {MODEL_PATH}")

    data = pd.read_csv(DATA_PATH)

    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"], errors="coerce"
    )
    data = data.drop(columns=["customerID"])
    data["Churn"] = data["Churn"].map({"No": 0, "Yes": 1})
    data = data.dropna(subset=["Churn"])

    X = data.drop(columns=["Churn"])
    y = data["Churn"].astype(int)

    # Keep the original training and test split consistent
    # with the baseline experiment.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Divide the test data into adaptation and final evaluation sets.
    # The final evaluation labels are not used for fitting.
    X_adapt, X_eval, y_adapt, y_eval = train_test_split(
        X_test,
        y_test,
        test_size=0.50,
        random_state=21,
        stratify=y_test,
    )

    # Apply synthetic drift to both subsets.
    X_adapt_shifted = apply_synthetic_drift(X_adapt)
    X_eval_shifted = apply_synthetic_drift(X_eval)

    baseline = load(MODEL_PATH)

    # Baseline performance on the shifted held-out evaluation set.
    baseline_metrics = calculate_metrics(
        baseline, X_eval_shifted, y_eval
    )

    # Clone the model's preprocessing and estimator pipeline.
    # Fit using original training data plus the shifted adaptation data.
    candidate = clone(baseline)

    X_retrain = pd.concat(
        [X_train, X_adapt_shifted],
        ignore_index=True,
    )
    y_retrain = pd.concat(
        [y_train, y_adapt.reset_index(drop=True)],
        ignore_index=True,
    )

    candidate.fit(X_retrain, y_retrain)

    # Evaluate the candidate on exactly the same held-out rows.
    candidate_metrics = calculate_metrics(
        candidate, X_eval_shifted, y_eval
    )

    rows = []

    for metric in baseline_metrics:
        base_value = baseline_metrics[metric]
        candidate_value = candidate_metrics[metric]

        rows.append({
            "metric": metric,
            "baseline": base_value,
            "retrained_candidate": candidate_value,
            "change_candidate_minus_baseline": (
                candidate_value - base_value
            ),
        })

    results = pd.DataFrame(rows)

    output_path = REPORT_DIR / "retraining_candidate_comparison.csv"
    results.to_csv(output_path, index=False)

    # Save the candidate separately; do not overwrite the baseline.
    candidate_path = ROOT / "models" / "retraining_candidate.joblib"

    from joblib import dump
    dump(candidate, candidate_path)

    print("\nRETRAINING CANDIDATE EVALUATION")
    print("=" * 90)
    print(f"Original training rows: {len(X_train)}")
    print(f"Adaptation rows: {len(X_adapt_shifted)}")
    print(f"Held-out evaluation rows: {len(X_eval_shifted)}")

    print("\nBaseline metrics on shifted evaluation data:")
    for metric, value in baseline_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nRetrained candidate metrics:")
    for metric, value in candidate_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nMetric comparison:")
    print(results.to_string(index=False))

    print(f"\nComparison saved to: {output_path}")
    print(f"Candidate model saved to: {candidate_path}")

    print("\nExperiment complete.")
    print(
        "Caution: synthetic shifts retain original labels; "
        "this is not evidence of production performance."
    )


if __name__ == "__main__":
    main()