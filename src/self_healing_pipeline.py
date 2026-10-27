
from pathlib import Path
import json
import warnings

import joblib
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

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
PSI_THRESHOLD = 0.25

# Synthetic drift settings used for this controlled experiment.
TENURE_MULTIPLIER = 1.5
MONTHLY_CHARGES_MULTIPLIER = 1.2

# Promotion requirements: the candidate must not regress
# on any of these metrics.
PROMOTION_METRICS = ["recall", "f1", "roc_auc"]


# ============================================================
# 2. LOAD AND PREPARE DATA
# ============================================================

def load_data():
    """Load the CSV and perform basic preparation."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Check that the CSV is inside data/raw/."
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = {
        "Churn",
        "customerID",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    # Convert TotalCharges from text to numeric.
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

    # Remove the identifier; it is not a useful predictive feature.
    df = df.drop(columns=["customerID"])

    # Encode target: No -> 0, Yes -> 1.
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    if df["Churn"].isna().any():
        raise ValueError("Unexpected values found in the Churn column.")

    df["Churn"] = df["Churn"].astype(int)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    return X, y


# ============================================================
# 3. BUILD THE MODEL PIPELINE
# ============================================================

def build_model(X):
    """Create a preprocessing and logistic regression pipeline."""

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
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

    return model


# ============================================================
# 4. PSI CALCULATION
# ============================================================

def calculate_numeric_psi(reference, current, bins=10):
    """
    Calculate PSI using bins defined from the reference data.
    Small smoothing prevents division by zero.
    """

    reference = pd.to_numeric(
        pd.Series(reference), errors="coerce"
    ).dropna()

    current = pd.to_numeric(
        pd.Series(current), errors="coerce"
    ).dropna()

    if len(reference) == 0 or len(current) == 0:
        return 0.0

    # Quantile-based reference bins.
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))

    # A constant reference feature cannot form multiple bins.
    if len(edges) < 2:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    epsilon = 1e-6

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    ref_pct = np.clip(ref_pct, epsilon, None)
    cur_pct = np.clip(cur_pct, epsilon, None)

    psi = np.sum(
        (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
    )

    return float(psi)


def calculate_categorical_psi(reference, current):
    """Calculate PSI for categorical distributions."""

    reference = pd.Series(reference).fillna("__MISSING__").astype(str)
    current = pd.Series(current).fillna("__MISSING__").astype(str)

    categories = sorted(set(reference.unique()) | set(current.unique()))

    if not categories:
        return 0.0

    ref_counts = reference.value_counts().reindex(
        categories, fill_value=0
    )
    cur_counts = current.value_counts().reindex(
        categories, fill_value=0
    )

    epsilon = 1e-6

    ref_pct = ref_counts.to_numpy(dtype=float) / len(reference)
    cur_pct = cur_counts.to_numpy(dtype=float) / len(current)

    ref_pct = np.clip(ref_pct, epsilon, None)
    cur_pct = np.clip(cur_pct, epsilon, None)

    psi = np.sum(
        (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
    )

    return float(psi)


def calculate_feature_drift(reference, current):
    """Calculate PSI for each original input feature."""

    rows = []

    for column in reference.columns:
        if pd.api.types.is_numeric_dtype(reference[column]):
            psi = calculate_numeric_psi(
                reference[column], current[column]
            )
        else:
            psi = calculate_categorical_psi(
                reference[column], current[column]
            )

        if psi < 0.10:
            level = "Low"
        elif psi < PSI_THRESHOLD:
            level = "Moderate"
        else:
            level = "High"

        rows.append(
            {
                "feature": column,
                "psi": psi,
                "drift_level": level,
            }
        )

    report = pd.DataFrame(rows).sort_values(
        "psi", ascending=False
    )

    return report


# ============================================================
# 5. CREATE A CONTROLLED SYNTHETIC DRIFT SCENARIO
# ============================================================

def apply_synthetic_drift(X):
    """
    Return a modified copy of X.
    This is a simulated covariate shift, not live data.
    """

    shifted = X.copy()

    shifted["tenure"] = (
        shifted["tenure"] * TENURE_MULTIPLIER
    ).clip(upper=72)

    shifted["MonthlyCharges"] = (
        shifted["MonthlyCharges"]
        * MONTHLY_CHARGES_MULTIPLIER
    ).clip(upper=120)

    return shifted


# ============================================================
# 6. MODEL EVALUATION
# ============================================================

def evaluate_model(model, X_eval, y_eval):
    """Calculate the evaluation metrics."""

    predictions = model.predict(X_eval)
    probabilities = model.predict_proba(X_eval)[:, 1]

    return {
        "accuracy": float(
            accuracy_score(y_eval, predictions)
        ),
        "precision": float(
            precision_score(
                y_eval, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_eval, predictions, zero_division=0
            )
        ),
        "f1": float(
            f1_score(y_eval, predictions, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_eval, probabilities)
        ),
    }


# ============================================================
# 7. PROMOTION GATE
# ============================================================

def promotion_gate(baseline_metrics, candidate_metrics):
    """
    Promote only if the candidate meets or exceeds the
    baseline on every configured promotion metric.
    """

    checks = {}

    for metric in PROMOTION_METRICS:
        baseline_value = baseline_metrics[metric]
        candidate_value = candidate_metrics[metric]

        checks[metric] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "delta": candidate_value - baseline_value,
            "passed": candidate_value >= baseline_value,
        }

    promote = all(
        result["passed"] for result in checks.values()
    )

    decision = "PROMOTE_CANDIDATE" if promote else "RETAIN_BASELINE"

    return decision, checks


# ============================================================
# 8. MAIN CLOSED-LOOP EXPERIMENT
# ============================================================

def main():
    print("=" * 65)
    print("SELF-HEALING ML: INTEGRATED EXPERIMENT")
    print("=" * 65)

    # --------------------------------------------------------
    # A. Load and split the dataset
    # --------------------------------------------------------

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Split the test portion into adaptation and final evaluation.
    # The evaluation partition is never used for fitting.
    X_adapt, X_eval, y_adapt, y_eval = train_test_split(
        X_test,
        y_test,
        test_size=0.50,
        random_state=21,
        stratify=y_test,
    )

    print(f"\nDataset rows: {len(X)}")
    print(f"Initial training rows: {len(X_train)}")
    print(f"Adaptation rows: {len(X_adapt)}")
    print(f"Final evaluation rows: {len(X_eval)}")

    # --------------------------------------------------------
    # B. Train baseline model on original training data
    # --------------------------------------------------------

    print("\n[1/7] Training baseline model...")

    baseline_model = build_model(X_train)
    baseline_model.fit(X_train, y_train)

    # --------------------------------------------------------
    # C. Simulate drift in adaptation and evaluation features
    # --------------------------------------------------------

    print("[2/7] Applying controlled synthetic drift...")

    X_adapt_shifted = apply_synthetic_drift(X_adapt)
    X_eval_shifted = apply_synthetic_drift(X_eval)

    # --------------------------------------------------------
    # D. Detect and diagnose drift
    # --------------------------------------------------------

    print("[3/7] Calculating feature-level PSI...")

    drift_report = calculate_feature_drift(
        X_train, X_adapt_shifted
    )

    drift_report.to_csv(
        REPORTS_DIR / "integrated_feature_drift_report.csv",
        index=False,
    )

    max_psi = float(drift_report["psi"].max())

    high_drift_features = drift_report.loc[
        drift_report["drift_level"] == "High", "feature"
    ].tolist()

    drift_detected = max_psi >= PSI_THRESHOLD

    print(f"Maximum PSI: {max_psi:.6f}")
    print(f"High-drift features: {high_drift_features}")

    # --------------------------------------------------------
    # E. Evaluate baseline on the final shifted evaluation set
    # --------------------------------------------------------

    print("[4/7] Evaluating baseline model...")

    baseline_metrics = evaluate_model(
        baseline_model, X_eval_shifted, y_eval
    )

    # --------------------------------------------------------
    # F. Consider and train a candidate model if drift is detected
    # --------------------------------------------------------

    candidate_metrics = None
    candidate_model = None
    gate_checks = {}
    decision = "SKIP_RETRAINING"
    retraining_considered = False

    if drift_detected:
        print("[5/7] Drift threshold reached. Training candidate...")

        retraining_considered = True

        # Fit on original training rows plus shifted adaptation rows.
        X_candidate_train = pd.concat(
            [X_train, X_adapt_shifted],
            ignore_index=True,
        )

        y_candidate_train = pd.concat(
            [y_train, y_adapt],
            ignore_index=True,
        )

        candidate_model = build_model(X_candidate_train)

        candidate_model.fit(
            X_candidate_train,
            y_candidate_train,
        )

        # Evaluate on the same held-out evaluation partition.
        candidate_metrics = evaluate_model(
            candidate_model, X_eval_shifted, y_eval
        )

        decision, gate_checks = promotion_gate(
            baseline_metrics, candidate_metrics
        )

    else:
        print("[5/7] Drift threshold not reached. Skipping retraining.")

    # --------------------------------------------------------
    # G. Save selected model and all experiment outputs
    # --------------------------------------------------------

    print("[6/7] Saving selected model and reports...")

    baseline_path = MODELS_DIR / "integrated_baseline_model.joblib"
    candidate_path = MODELS_DIR / "integrated_candidate_model.joblib"
    selected_path = MODELS_DIR / "integrated_selected_model.joblib"

    joblib.dump(baseline_model, baseline_path)

    if candidate_model is not None:
        joblib.dump(candidate_model, candidate_path)

    if decision == "PROMOTE_CANDIDATE":
        selected_model = candidate_model
        selected_model_name = "candidate"
    else:
        selected_model = baseline_model
        selected_model_name = "baseline"

    joblib.dump(selected_model, selected_path)

    comparison_rows = [
        {"model": "baseline", **baseline_metrics}
    ]

    if candidate_metrics is not None:
        comparison_rows.append(
            {"model": "candidate", **candidate_metrics}
        )

    comparison_df = pd.DataFrame(comparison_rows)

    comparison_df.to_csv(
        REPORTS_DIR / "integrated_model_comparison.csv",
        index=False,
    )

    summary = {
        "experiment": "Integrated Self-Healing ML Pipeline",
        "dataset": DATA_PATH.name,
        "scenario": "controlled_synthetic_covariate_shift",
        "random_state": RANDOM_STATE,
        "psi_threshold": PSI_THRESHOLD,
        "synthetic_drift": {
            "tenure_multiplier": TENURE_MULTIPLIER,
            "monthly_charges_multiplier": MONTHLY_CHARGES_MULTIPLIER,
        },
        "partition_sizes": {
            "training": int(len(X_train)),
            "adaptation": int(len(X_adapt)),
            "evaluation": int(len(X_eval)),
        },
        "max_psi": max_psi,
        "high_drift_features": high_drift_features,
        "drift_detected": bool(drift_detected),
        "retraining_considered": bool(retraining_considered),
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "promotion_checks": gate_checks,
        "final_decision": decision,
        "selected_model": selected_model_name,
        "limitations": [
            "Drift is synthetically generated, not observed in production.",
            "The original labels are retained after feature modification.",
            "PSI detects distribution change, not proof of performance degradation.",
            "The promotion gate uses recall, F1, and ROC-AUC only.",
            "Results apply to this dataset, split, and synthetic scenario.",
        ],
    }

    summary_path = REPORTS_DIR / "integrated_pipeline_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    # Human-readable report.
    report_lines = [
        "SELF-HEALING ML — INTEGRATED PIPELINE REPORT",
        "=" * 55,
        f"Scenario: {summary['scenario']}",
        f"Maximum PSI: {max_psi:.6f}",
        f"PSI threshold: {PSI_THRESHOLD}",
        f"High-drift features: {high_drift_features}",
        f"Drift detected: {drift_detected}",
        f"Retraining considered: {retraining_considered}",
        "",
        "BASELINE METRICS",
    ]

    for metric, value in baseline_metrics.items():
        report_lines.append(f"{metric}: {value:.4f}")

    if candidate_metrics is not None:
        report_lines.extend(["", "CANDIDATE METRICS"])

        for metric, value in candidate_metrics.items():
            report_lines.append(f"{metric}: {value:.4f}")

        report_lines.extend(["", "PROMOTION GATE"])

        for metric, result in gate_checks.items():
            status = "PASS" if result["passed"] else "FAIL"
            report_lines.append(
                f"{metric}: {status} "
                f"(baseline={result['baseline']:.4f}, "
                f"candidate={result['candidate']:.4f}, "
                f"delta={result['delta']:+.4f})"
            )

    report_lines.extend(
        [
            "",
            f"FINAL DECISION: {decision}",
            f"SELECTED MODEL: {selected_model_name}",
            "",
            "LIMITATIONS",
            *[f"- {item}" for item in summary["limitations"]],
        ]
    )

    with open(
        REPORTS_DIR / "integrated_pipeline_report.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write("\n".join(report_lines))

    # --------------------------------------------------------
    # H. Print final summary
    # --------------------------------------------------------

    print("[7/7] Pipeline completed.")

    print("\n" + "=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)

    print(f"Drift detected: {drift_detected}")
    print(f"Maximum PSI: {max_psi:.6f}")
    print(f"High-drift features: {high_drift_features}")

    print("\nBaseline metrics:")
    for metric, value in baseline_metrics.items():
        print(f"  {metric}: {value:.4f}")

    if candidate_metrics is not None:
        print("\nCandidate metrics:")
        for metric, value in candidate_metrics.items():
            print(f"  {metric}: {value:.4f}")

        print("\nPromotion gate:")
        for metric, result in gate_checks.items():
            status = "PASS" if result["passed"] else "FAIL"
            print(f"  {metric}: {status}")

    print(f"\nFINAL DECISION: {decision}")
    print(f"Selected model: {selected_model_name}")

    print("\nSaved files:")
    print(f"  {baseline_path}")
    if candidate_model is not None:
        print(f"  {candidate_path}")
    print(f"  {selected_path}")
    print(f"  {REPORTS_DIR / 'integrated_feature_drift_report.csv'}")
    print(f"  {REPORTS_DIR / 'integrated_model_comparison.csv'}")
    print(f"  {summary_path}")
    print(f"  {REPORTS_DIR / 'integrated_pipeline_report.txt'}")


if __name__ == "__main__":
    main()