
from pathlib import Path

import numpy as np
import pandas as pd

from joblib import load, dump
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
MODEL_DIR = ROOT / "models"

SEEDS = [11, 21, 42, 55, 77]


def apply_synthetic_drift(frame):
    """Apply the stronger synthetic feature shift."""
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
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Baseline pipeline not found: {MODEL_PATH}")

    data = pd.read_csv(DATA_PATH)

    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"], errors="coerce"
    )
    data = data.drop(columns=["customerID"])
    data["Churn"] = data["Churn"].map({"No": 0, "Yes": 1})
    data = data.dropna(subset=["Churn"])

    X = data.drop(columns=["Churn"])
    y = data["Churn"].astype(int)

    # Load the saved pipeline as a model template.
    # Each seed gets fresh, independently fitted models.
    model_template = load(MODEL_PATH)

    all_rows = []

    for seed in SEEDS:
        print(f"\nRunning random seed: {seed}")

        # First split: training data and held-out test data.
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=seed,
            stratify=y,
        )

        # Second split: adaptation data and final evaluation data.
        X_adapt, X_eval, y_adapt, y_eval = train_test_split(
            X_test,
            y_test,
            test_size=0.50,
            random_state=seed + 100,
            stratify=y_test,
        )

        # Apply synthetic shift to the adaptation and evaluation features.
        X_adapt_shifted = apply_synthetic_drift(X_adapt)
        X_eval_shifted = apply_synthetic_drift(X_eval)

        # Fit a fresh baseline using only the original training subset.
        baseline = clone(model_template)
        baseline.fit(X_train, y_train)

        baseline_metrics = calculate_metrics(
            baseline, X_eval_shifted, y_eval
        )

        # Fit a fresh candidate using original training data plus
        # the shifted adaptation data. Evaluation data is not used.
        candidate = clone(model_template)

        X_retrain = pd.concat(
            [X_train, X_adapt_shifted],
            ignore_index=True,
        )
        y_retrain = pd.concat(
            [
                y_train.reset_index(drop=True),
                y_adapt.reset_index(drop=True),
            ],
            ignore_index=True,
        )

        candidate.fit(X_retrain, y_retrain)

        candidate_metrics = calculate_metrics(
            candidate, X_eval_shifted, y_eval
        )

        row = {
            "seed": seed,
            "training_rows": len(X_train),
            "adaptation_rows": len(X_adapt_shifted),
            "evaluation_rows": len(X_eval_shifted),
        }

        for metric in baseline_metrics:
            row[f"baseline_{metric}"] = baseline_metrics[metric]
            row[f"candidate_{metric}"] = candidate_metrics[metric]
            row[f"change_{metric}"] = (
                candidate_metrics[metric] - baseline_metrics[metric]
            )

        all_rows.append(row)

    results = pd.DataFrame(all_rows)

    results_path = REPORT_DIR / "multi_seed_stability_results.csv"
    results.to_csv(results_path, index=False)

    # Summarize mean and standard deviation of each candidate-minus-baseline
    # difference across the selected random seeds.
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    summary_rows = []

    for metric in metric_names:
        changes = results[f"change_{metric}"]

        summary_rows.append({
            "metric": metric,
            "mean_change": changes.mean(),
            "std_change": changes.std(ddof=1),
            "seeds_candidate_better": int((changes > 0).sum()),
            "seeds_baseline_better": int((changes < 0).sum()),
            "seeds_tied": int((changes == 0).sum()),
        })

    summary = pd.DataFrame(summary_rows)

    summary_path = REPORT_DIR / "multi_seed_stability_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nMULTI-SEED RETRAINING STABILITY EXPERIMENT")
    print("=" * 100)

    print("\nPer-seed metric changes (candidate minus baseline):")
    change_columns = [
        "seed",
        "change_accuracy",
        "change_precision",
        "change_recall",
        "change_f1",
        "change_roc_auc",
    ]
    print(results[change_columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nSummary across seeds:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\nDetailed results saved to: {results_path}")
    print(f"Summary saved to: {summary_path}")

    print("\nExperiment complete.")
    print(
        "Interpret results as stability evidence across these splits only. "
        "All feature shifts are synthetic, and the original labels are retained."
    )


if __name__ == "__main__":
    main()