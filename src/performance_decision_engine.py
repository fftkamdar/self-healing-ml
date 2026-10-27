
from pathlib import Path
import json

import pandas as pd

from self_healing_pipeline import (
    PSI_THRESHOLD,
    PROMOTION_METRICS,
    REPORTS_DIR,
)


# ============================================================
# 1. DECISION ENGINE
# ============================================================

def make_decision(
    drift_report,
    baseline_metrics,
    candidate_metrics=None,
):
    """
    Make a retraining decision using:
    1. Feature-level PSI
    2. Baseline performance
    3. Candidate performance, when available

    This is a decision-support rule, not proof that
    retraining is necessary or that drift caused degradation.
    """

    max_psi = float(drift_report["psi"].max())

    high_features = drift_report.loc[
        drift_report["psi"] >= PSI_THRESHOLD,
        "feature",
    ].tolist()

    moderate_features = drift_report.loc[
        (drift_report["psi"] >= 0.10)
        & (drift_report["psi"] < PSI_THRESHOLD),
        "feature",
    ].tolist()

    drift_detected = max_psi >= PSI_THRESHOLD

    # --------------------------------------------------------
    # Case 1: Drift is below threshold
    # --------------------------------------------------------

    if not drift_detected:
        return {
            "decision": "MONITOR",
            "max_psi": max_psi,
            "drift_detected": False,
            "high_drift_features": high_features,
            "moderate_drift_features": moderate_features,
            "promotion_checks": {},
            "reasons": [
                "Maximum PSI is below the configured threshold.",
                "No candidate evaluation is required by this rule.",
                "Continue monitoring subsequent data batches.",
            ],
        }

    # --------------------------------------------------------
    # Case 2: Drift detected, candidate not yet evaluated
    # --------------------------------------------------------

    if candidate_metrics is None:
        return {
            "decision": "EVALUATE_CANDIDATE",
            "max_psi": max_psi,
            "drift_detected": True,
            "high_drift_features": high_features,
            "moderate_drift_features": moderate_features,
            "promotion_checks": {},
            "reasons": [
                "At least one feature has PSI at or above the threshold.",
                "A candidate should be evaluated on a held-out labelled set.",
                "Drift alone is not sufficient evidence to promote a model.",
            ],
        }

    # --------------------------------------------------------
    # Case 3: Candidate metrics are available
    # --------------------------------------------------------

    checks = {}

    for metric in PROMOTION_METRICS:
        baseline_value = float(baseline_metrics[metric])
        candidate_value = float(candidate_metrics[metric])
        delta = candidate_value - baseline_value

        checks[metric] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "delta": delta,
            "passed": candidate_value >= baseline_value,
        }

    all_passed = all(
        result["passed"] for result in checks.values()
    )

    if all_passed:
        decision = "PROMOTE_CANDIDATE"
        reasons = [
            "Drift was detected.",
            "A candidate was evaluated against the baseline.",
            "The candidate met or exceeded the baseline on every configured promotion metric.",
        ]
    else:
        decision = "RETAIN_BASELINE"
        failed_metrics = [
            metric
            for metric, result in checks.items()
            if not result["passed"]
        ]

        reasons = [
            "Drift was detected and a candidate was evaluated.",
            "The candidate failed one or more configured promotion checks.",
            f"Failed checks: {', '.join(failed_metrics)}.",
            "Retain the baseline under the current promotion policy.",
        ]

    return {
        "decision": decision,
        "max_psi": max_psi,
        "drift_detected": True,
        "high_drift_features": high_features,
        "moderate_drift_features": moderate_features,
        "promotion_checks": checks,
        "reasons": reasons,
    }


# ============================================================
# 2. DEMONSTRATION USING SAVED INTEGRATED REPORTS
# ============================================================

def main():
    print("=" * 70)
    print("PERFORMANCE-AWARE RETRAINING DECISION ENGINE")
    print("=" * 70)

    drift_path = REPORTS_DIR / "integrated_feature_drift_report.csv"
    summary_path = REPORTS_DIR / "integrated_pipeline_summary.json"

    if not drift_path.exists():
        raise FileNotFoundError(
            f"Required report not found: {drift_path}\n"
            "Run src/self_healing_pipeline.py first."
        )

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Required summary not found: {summary_path}\n"
            "Run src/self_healing_pipeline.py first."
        )

    drift_report = pd.read_csv(drift_path)

    with open(summary_path, "r", encoding="utf-8") as file:
        summary = json.load(file)

    baseline_metrics = summary["baseline_metrics"]
    candidate_metrics = summary.get("candidate_metrics")

    result = make_decision(
        drift_report,
        baseline_metrics,
        candidate_metrics,
    )

    output_path = REPORTS_DIR / "performance_decision.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)

    print(f"\nDecision: {result['decision']}")
    print(f"Maximum PSI: {result['max_psi']:.6f}")
    print(f"Drift detected: {result['drift_detected']}")
    print(
        "High-drift features:",
        result["high_drift_features"],
    )

    if result["promotion_checks"]:
        print("\nPromotion checks:")

        for metric, check in result["promotion_checks"].items():
            status = "PASS" if check["passed"] else "FAIL"

            print(
                f"  {metric}: {status} | "
                f"baseline={check['baseline']:.4f} | "
                f"candidate={check['candidate']:.4f} | "
                f"delta={check['delta']:+.4f}"
            )

    print("\nReasons:")

    for reason in result["reasons"]:
        print(f"- {reason}")

    print(f"\nSaved decision report: {output_path}")


if __name__ == "__main__":
    main()