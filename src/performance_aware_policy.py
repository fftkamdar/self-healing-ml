
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from joblib import load


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
MODEL_PATH = ROOT / "models" / "baseline_churn_pipeline.joblib"
REPORT_DIR = ROOT / "reports"

PSI_THRESHOLD = 0.25

# Example guardrails for this experiment.
# These are experimental settings, not universal production standards.
F1_DROP_GUARDRAIL = 0.03
ROC_AUC_DROP_GUARDRAIL = 0.02


def calculate_psi(reference, current, bins=10):
    """Calculate PSI for numeric data using reference-distribution bins."""
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()

    if len(reference) == 0 or len(current) == 0:
        return 0.0

    edges = np.unique(
        np.quantile(reference, np.linspace(0, 1, bins + 1))
    )

    if len(edges) < 2:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts = np.histogram(reference, bins=edges)[0]
    cur_counts = np.histogram(current, bins=edges)[0]

    ref_pct = ref_counts / len(reference)
    cur_pct = cur_counts / len(current)

    epsilon = 1e-6
    ref_pct = np.clip(ref_pct, epsilon, None)
    cur_pct = np.clip(cur_pct, epsilon, None)

    return float(np.sum(
        (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
    ))


def apply_synthetic_drift(frame, tenure_multiplier, charges_multiplier):
    """Apply controlled changes to two numeric features."""
    shifted = frame.copy()

    shifted["tenure"] = (
        pd.to_numeric(shifted["tenure"], errors="coerce")
        * tenure_multiplier
    ).clip(upper=72)

    shifted["MonthlyCharges"] = (
        pd.to_numeric(shifted["MonthlyCharges"], errors="coerce")
        * charges_multiplier
    ).clip(upper=120)

    return shifted


def evaluate(model, X, y):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
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

    # Keep a held-out test set separate from the training data.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = load(MODEL_PATH)

    # Reference performance on the unmodified held-out test set.
    reference_metrics = evaluate(model, X_test, y_test)

    scenarios = [
        ("No shift", 1.0, 1.0),
        ("Mild shift", 1.1, 1.05),
        ("Moderate shift", 1.3, 1.1),
        ("Stronger shift", 1.5, 1.2),
    ]

    rows = []

    for name, tenure_mult, charges_mult in scenarios:
        X_current = apply_synthetic_drift(
            X_test, tenure_mult, charges_mult
        )

        psi_scores = {}

        for column in X_train.columns:
            if pd.api.types.is_numeric_dtype(X_train[column]):
                psi_scores[column] = calculate_psi(
                    X_train[column], X_current[column]
                )

        max_psi = max(psi_scores.values(), default=0.0)
        high_drift_features = [
            feature
            for feature, score in psi_scores.items()
            if score >= PSI_THRESHOLD
        ]

        current_metrics = evaluate(model, X_current, y_test)

        f1_drop = reference_metrics["f1"] - current_metrics["f1"]
        auc_drop = (
            reference_metrics["roc_auc"] - current_metrics["roc_auc"]
        )

        drift_detected = max_psi >= PSI_THRESHOLD
        performance_breached = (
            f1_drop >= F1_DROP_GUARDRAIL
            or auc_drop >= ROC_AUC_DROP_GUARDRAIL
        )

        if not drift_detected:
            decision = "SKIP_RETRAINING"
        elif performance_breached:
            decision = "RETRAINING_CANDIDATE"
        else:
            decision = "MONITOR_NO_AUTOMATIC_RETRAINING"

        rows.append({
            "scenario": name,
            "tenure_multiplier": tenure_mult,
            "monthly_charges_multiplier": charges_mult,
            "max_psi": max_psi,
            "high_drift_features": ", ".join(high_drift_features),
            "reference_f1": reference_metrics["f1"],
            "current_f1": current_metrics["f1"],
            "f1_drop": f1_drop,
            "reference_roc_auc": reference_metrics["roc_auc"],
            "current_roc_auc": current_metrics["roc_auc"],
            "roc_auc_drop": auc_drop,
            "drift_detected": drift_detected,
            "performance_guardrail_breached": performance_breached,
            "decision": decision,
        })

    results = pd.DataFrame(rows)
    output_path = REPORT_DIR / "performance_aware_policy_results.csv"
    results.to_csv(output_path, index=False)

    summary = {
        "psi_threshold": PSI_THRESHOLD,
        "f1_drop_guardrail": F1_DROP_GUARDRAIL,
        "roc_auc_drop_guardrail": ROC_AUC_DROP_GUARDRAIL,
        "reference_metrics": reference_metrics,
        "results": rows,
        "limitations": [
            "All feature shifts are synthetic.",
            "Original test labels are retained after feature transformations.",
            "The guardrail values are experimental choices.",
            "A retraining candidate must be evaluated separately before deployment.",
        ],
    }

    summary_path = REPORT_DIR / "performance_aware_policy_summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\nPERFORMANCE-AWARE RETRAINING POLICY")
    print("=" * 110)
    print("Reference performance:", reference_metrics)
    print("\nScenario results:")
    print(results.to_string(index=False))
    print(f"\nSaved CSV: {output_path}")
    print(f"Saved JSON: {summary_path}")
    print("\nExperiment complete.")
    print("Synthetic shifts and retained labels limit real-world interpretation.")


if __name__ == "__main__":
    main()