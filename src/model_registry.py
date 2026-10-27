
"""
Model Registry and Rollback Simulation
--------------------------------------
Registers baseline and candidate model versions, records their
evaluation metrics, and simulates rollback when a candidate
does not pass the promotion gate.

This is a local experimental registry, not a production model
management service.
"""

import json
from datetime import datetime, timezone

import joblib
import pandas as pd

from self_healing_pipeline import MODELS_DIR, REPORTS_DIR


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASELINE_FILENAME = "integrated_baseline_model.joblib"
CANDIDATE_FILENAME = "integrated_candidate_model.joblib"
SELECTED_FILENAME = "integrated_selected_model.joblib"

SUMMARY_FILENAME = "integrated_pipeline_summary.json"

REGISTRY_FILENAME = "model_registry.json"
ROLLBACK_REPORT_FILENAME = "rollback_simulation_report.txt"

# The integrated pipeline currently uses these metrics
# as its promotion requirements.
REQUIRED_METRICS = ["recall", "f1", "roc_auc"]


# ============================================================
# 2. FILE AND JSON HELPERS
# ============================================================

def load_json(path):
    """Load a JSON file and return its contents."""

    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def find_model_path(filename):
    """Return the model path or raise an informative error."""

    path = MODELS_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {path}\n"
            "Run src/self_healing_pipeline.py first."
        )

    return path


def extract_metrics(summary, model_name):
    """
    Extract metrics from the integrated pipeline summary.

    Supports the common nested dictionary layout and raises
    a clear error if the expected values cannot be located.
    """

    possible_keys = [
        model_name,
        f"{model_name}_metrics",
        f"{model_name}_results",
    ]

    for key in possible_keys:
        value = summary.get(key)

        if isinstance(value, dict):
            if all(metric in value for metric in REQUIRED_METRICS):
                return {
                    metric: float(value[metric])
                    for metric in REQUIRED_METRICS
                }

    # Also support summaries where metrics are nested under
    # a "model_comparison" or "comparison" object.
    for parent_key in ["model_comparison", "comparison", "metrics"]:
        parent = summary.get(parent_key)

        if isinstance(parent, dict):
            for key in possible_keys:
                value = parent.get(key)

                if isinstance(value, dict):
                    if all(metric in value for metric in REQUIRED_METRICS):
                        return {
                            metric: float(value[metric])
                            for metric in REQUIRED_METRICS
                        }

    raise KeyError(
        f"Could not locate {model_name} metrics in {SUMMARY_FILENAME}. "
        "Inspect the summary JSON and adjust extract_metrics() "
        "to match its structure."
    )


def determine_promotion_status(summary):
    """
    Read the promotion gate decision from the integrated summary.

    Returns a boolean and a descriptive explanation.
    """

    possible_keys = [
        "promotion_decision",
        "promotion_status",
        "decision",
        "final_decision",
    ]

    for key in possible_keys:
        value = summary.get(key)

        if isinstance(value, str):
            normalized = value.strip().upper()

            if "PROMOTE" in normalized:
                return True, value

            if "RETAIN_BASELINE" in normalized:
                return False, value

    # If a direct decision string is not available, evaluate
    # candidate versus baseline on all required metrics.
    baseline = extract_metrics(summary, "baseline")
    candidate = extract_metrics(summary, "candidate")

    passed = all(
        candidate[metric] >= baseline[metric]
        for metric in REQUIRED_METRICS
    )

    return passed, (
        "Promotion gate recomputed from required metrics: "
        + ("PASS" if passed else "FAIL")
    )


# ============================================================
# 3. REGISTRY CREATION
# ============================================================

def create_model_record(
    version,
    model_name,
    artifact_path,
    metrics,
    status,
):
    """Create a registry entry for one model version."""

    return {
        "version": version,
        "model_name": model_name,
        "artifact": str(artifact_path),
        "metrics": metrics,
        "status": status,
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def save_registry(registry):
    """Save registry information to JSON."""

    registry_path = REPORTS_DIR / REGISTRY_FILENAME

    with open(registry_path, "w", encoding="utf-8") as file:
        json.dump(registry, file, indent=4)

    return registry_path


# ============================================================
# 4. ROLLBACK SIMULATION
# ============================================================

def simulate_rollback(
    baseline_path,
    candidate_path,
    selected_path,
    candidate_promoted,
):
    """
    Simulate selecting the candidate if it passes the gate;
    otherwise select the baseline.

    The source artifacts are not modified.
    """

    if candidate_promoted:
        chosen_path = candidate_path
        selected_version = "v2"
        decision = "PROMOTE_CANDIDATE"
        explanation = (
            "The candidate passed the configured promotion gate. "
            "The candidate is selected."
        )
    else:
        chosen_path = baseline_path
        selected_version = "v1"
        decision = "ROLLBACK_TO_BASELINE"
        explanation = (
            "The candidate did not pass the configured promotion gate. "
            "The baseline is retained as the selected model."
        )

    # Load the chosen artifact to verify it can be deserialized.
    selected_model = joblib.load(chosen_path)

    # Save a separate selected-model artifact.
    joblib.dump(selected_model, selected_path)

    return {
        "decision": decision,
        "selected_version": selected_version,
        "selected_artifact": str(selected_path),
        "source_artifact": str(chosen_path),
        "explanation": explanation,
    }


# ============================================================
# 5. MAIN
# ============================================================

def main():
    print("=" * 70)
    print("MODEL REGISTRY AND ROLLBACK SIMULATION")
    print("=" * 70)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    baseline_path = find_model_path(BASELINE_FILENAME)
    candidate_path = find_model_path(CANDIDATE_FILENAME)
    selected_path = MODELS_DIR / SELECTED_FILENAME

    summary_path = REPORTS_DIR / SUMMARY_FILENAME
    summary = load_json(summary_path)

    baseline_metrics = extract_metrics(summary, "baseline")
    candidate_metrics = extract_metrics(summary, "candidate")

    candidate_promoted, gate_reason = determine_promotion_status(summary)

    print("\nBaseline metrics:")
    for metric, value in baseline_metrics.items():
        print(f"  {metric}: {value:.4f}")

    print("\nCandidate metrics:")
    for metric, value in candidate_metrics.items():
        print(f"  {metric}: {value:.4f}")

    print("\nPromotion gate:")
    print(f"  {gate_reason}")

    # --------------------------------------------------------
    # Register versions
    # --------------------------------------------------------

    baseline_status = "ACTIVE" if not candidate_promoted else "PREVIOUS"
    candidate_status = "ACTIVE" if candidate_promoted else "REJECTED"

    registry = {
        "registry_name": "Self-Healing ML Local Model Registry",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_version": "v2" if candidate_promoted else "v1",
        "promotion_requirements": REQUIRED_METRICS,
        "models": [
            create_model_record(
                version="v1",
                model_name="Logistic Regression Baseline",
                artifact_path=baseline_path,
                metrics=baseline_metrics,
                status=baseline_status,
            ),
            create_model_record(
                version="v2",
                model_name="Logistic Regression Candidate",
                artifact_path=candidate_path,
                metrics=candidate_metrics,
                status=candidate_status,
            ),
        ],
    }

    registry_path = save_registry(registry)

    # --------------------------------------------------------
    # Simulate selection or rollback
    # --------------------------------------------------------

    rollback_result = simulate_rollback(
        baseline_path=baseline_path,
        candidate_path=candidate_path,
        selected_path=selected_path,
        candidate_promoted=candidate_promoted,
    )

    # --------------------------------------------------------
    # Write report
    # --------------------------------------------------------

    report_path = REPORTS_DIR / ROLLBACK_REPORT_FILENAME

    report_lines = [
        "MODEL REGISTRY AND ROLLBACK SIMULATION",
        "=" * 70,
        "",
        f"Promotion gate result: {gate_reason}",
        f"Final action: {rollback_result['decision']}",
        f"Selected model version: {rollback_result['selected_version']}",
        f"Selected artifact: {rollback_result['selected_artifact']}",
        "",
        "Baseline metrics:",
    ]

    report_lines.extend(
        f"  {metric}: {value:.4f}"
        for metric, value in baseline_metrics.items()
    )

    report_lines.extend(["", "Candidate metrics:"])

    report_lines.extend(
        f"  {metric}: {value:.4f}"
        for metric, value in candidate_metrics.items()
    )

    report_lines.extend([
        "",
        "Explanation:",
        rollback_result["explanation"],
        "",
        "Limitations:",
        "- This is a local file-based registry simulation.",
        "- It does not provide production access control or audit infrastructure.",
        "- The selected model is copied to a separate artifact path.",
        "- Promotion depends on the configured metrics and integrated summary.",
        "- This experiment does not establish production readiness.",
        "",
        f"Registry JSON: {registry_path}",
        f"Selected model artifact: {selected_path}",
    ])

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\nRegistry entries:")
    for model in registry["models"]:
        print(
            f"  {model['version']} | {model['model_name']} | "
            f"{model['status']}"
        )

    print("\nRollback simulation:")
    print(f"  Decision: {rollback_result['decision']}")
    print(f"  Selected version: {rollback_result['selected_version']}")
    print(f"  Explanation: {rollback_result['explanation']}")

    print("\nSaved outputs:")
    print(registry_path)
    print(report_path)
    print(selected_path)


if __name__ == "__main__":
    main()