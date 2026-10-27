
from pathlib import Path
import json

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
BASELINE_PATH = ROOT / "models" / "baseline_churn_pipeline.joblib"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"

RANDOM_STATE = 42

# Illustrative promotion criteria.
# Candidate must meet or exceed the baseline on all three metrics.
REQUIRED_METRICS = ["recall", "f1", "roc_auc"]


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
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
    }


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    if not BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"Baseline model not found: {BASELINE_PATH}"
        )

    # Load and prepare the dataset.
    data = pd.read_csv(DATA_PATH)

    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"], errors="coerce"
    )

    data = data.drop(columns=["customerID"])
    data["Churn"] = data["Churn"].map({"No": 0, "Yes": 1})
    data = data.dropna(subset=["Churn"])

    X = data.drop(columns=["Churn"])
    y = data["Churn"].astype(int)

    # Reproduce the split used by the existing baseline model.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Separate adaptation data from final evaluation data.
    X_adapt, X_eval, y_adapt, y_eval = train_test_split(
        X_test,
        y_test,
        test_size=0.50,
        random_state=21,
        stratify=y_test,
    )

    # Apply the same synthetic shift to both partitions.
    X_adapt_shifted = apply_synthetic_drift(X_adapt)
    X_eval_shifted = apply_synthetic_drift(X_eval)

    # Load the existing baseline.
    baseline = load(BASELINE_PATH)

    baseline_metrics = calculate_metrics(
        baseline, X_eval_shifted, y_eval
    )

    # Create a candidate with the same pipeline structure.
    candidate = clone(baseline)

    X_candidate_train = pd.concat(
        [X_train, X_adapt_shifted],
        ignore_index=True,
    )

    y_candidate_train = pd.concat(
        [
            y_train.reset_index(drop=True),
            y_adapt.reset_index(drop=True),
        ],
        ignore_index=True,
    )

    candidate.fit(X_candidate_train, y_candidate_train)

    candidate_metrics = calculate_metrics(
        candidate, X_eval_shifted, y_eval
    )

    # Check each required metric.
    gate_results = {}

    for metric in REQUIRED_METRICS:
        gate_results[metric] = {
            "baseline": baseline_metrics[metric],
            "candidate": candidate_metrics[metric],
            "passed": (
                candidate_metrics[metric] >= baseline_metrics[metric]
            ),
        }

    all_gates_passed = all(
        result["passed"] for result in gate_results.values()
    )

    if all_gates_passed:
        decision = "PROMOTE_CANDIDATE"
        selected_model = candidate
        selected_model_path = MODEL_DIR / "promoted_churn_pipeline.joblib"
    else:
        decision = "RETAIN_BASELINE"
        selected_model = baseline
        selected_model_path = MODEL_DIR / "selected_churn_pipeline.joblib"

    # Save the selected model separately; never overwrite the baseline.
    dump(selected_model, selected_model_path)

    # Save a full comparison report.
    comparison_rows = []

    for metric in baseline_metrics:
        comparison_rows.append({
            "metric": metric,
            "baseline": baseline_metrics[metric],
            "candidate": candidate_metrics[metric],
            "candidate_minus_baseline": (
                candidate_metrics[metric] - baseline_metrics[metric]
            ),
            "required_for_promotion": metric in REQUIRED_METRICS,
            "gate_passed": (
                gate_results[metric]["passed"]
                if metric in gate_results
                else None
            ),
        })

    comparison = pd.DataFrame(comparison_rows)

    comparison_path = REPORT_DIR / "model_promotion_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    summary = {
        "experiment": "Model Promotion Gate",
        "random_state": RANDOM_STATE,
        "training_rows": len(X_train),
        "adaptation_rows": len(X_adapt_shifted),
        "evaluation_rows": len(X_eval_shifted),
        "required_metrics": REQUIRED_METRICS,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "gate_results": gate_results,
        "all_gates_passed": all_gates_passed,
        "decision": decision,
        "selected_model_path": str(selected_model_path),
        "limitations": [
            "Feature transformations are synthetic.",
            "Original labels are retained after transformation.",
            "Promotion criteria are illustrative and not production standards.",
            "Evaluation uses one held-out partition.",
            "This experiment does not establish real-world model performance.",
        ],
    }

    summary_path = REPORT_DIR / "model_promotion_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\nMODEL PROMOTION GATE")
    print("=" * 85)

    print("\nBaseline metrics:")
    for metric, value in baseline_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nCandidate metrics:")
    for metric, value in candidate_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nPromotion gate checks:")
    for metric, result in gate_results.items():
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"{metric}: {status} "
            f"(baseline={result['baseline']:.4f}, "
            f"candidate={result['candidate']:.4f})"
        )

    print(f"\nFinal decision: {decision}")
    print(f"Selected model saved to: {selected_model_path}")
    print(f"Comparison report: {comparison_path}")
    print(f"Summary report: {summary_path}")

    print("\nExperiment complete.")
    print(
        "The original baseline model was not overwritten. "
        "Results are limited to this synthetic experiment."
    )


if __name__ == "__main__":
    main()