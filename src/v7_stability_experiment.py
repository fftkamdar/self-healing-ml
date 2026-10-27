"""
V7 Stability Experiment

Runs the existing SH-DAS V7 experiment across multiple random seeds
and produces:

1. Per-seed results CSV
2. Mean / standard deviation summary CSV
3. JSON summary

The V7 experiment itself is not modified.
Only RANDOM_STATE and REPORT_DIR are changed for each run.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

V7_SCRIPT = (
    PROJECT_ROOT
    / "src"
    / "proposed_model_v7.py"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "proposed_model_v7"
    / "stability"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

SEEDS = [
    11,
    21,
    42,
    55,
    77,
]


# ============================================================================
# IMPORT V7 MODULE
# ============================================================================

def load_v7_module():

    spec = importlib.util.spec_from_file_location(
        "proposed_model_v7",
        V7_SCRIPT,
    )

    if spec is None or spec.loader is None:

        raise ImportError(
            f"Could not load V7 script: {V7_SCRIPT}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


# ============================================================================
# RUN ONE SEED
# ============================================================================

def run_one_seed(
    seed: int,
) -> dict:

    print()
    print("=" * 75)
    print(
        f"RUNNING SH-DAS V7 — SEED {seed}"
    )
    print("=" * 75)

    v7 = load_v7_module()

    # ------------------------------------------------------------------------
    # Change only the random seed.
    # ------------------------------------------------------------------------

    v7.RANDOM_STATE = seed

    # ------------------------------------------------------------------------
    # Save each seed's artifacts separately.
    # ------------------------------------------------------------------------

    seed_report_dir = (
        OUTPUT_DIR
        / f"seed_{seed}"
    )

    seed_report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    v7.REPORT_DIR = seed_report_dir

    # ------------------------------------------------------------------------
    # Suppress the very long V7 console output.
    # ------------------------------------------------------------------------

    captured_output = io.StringIO()

    try:

        with contextlib.redirect_stdout(
            captured_output
        ):

            v7.main()

    except Exception as exc:

        print(
            f"Seed {seed} FAILED: {exc}"
        )

        return {
            "Seed": seed,
            "Status": "FAILED",
            "Error": str(exc),
        }

    # ------------------------------------------------------------------------
    # Read the final V7 comparison CSV.
    # ------------------------------------------------------------------------

    comparison_path = (
        seed_report_dir
        / "six_model_v7_final_comparison.csv"
    )

    if not comparison_path.exists():

        return {
            "Seed": seed,
            "Status": "FAILED",
            "Error":
                "Final comparison CSV was not created.",
        }

    comparison = pd.read_csv(
        comparison_path
    )

    proposed = comparison[
        comparison["Model"]
        == "SH-DAS V7"
    ]

    if proposed.empty:

        return {
            "Seed": seed,
            "Status": "FAILED",
            "Error":
                "SH-DAS V7 row not found.",
        }

    row = proposed.iloc[0]

    # ------------------------------------------------------------------------
    # Return only the important metrics.
    # ------------------------------------------------------------------------

    return {
        "Seed": seed,
        "Status": "SUCCESS",
        "Accuracy": float(
            row["Accuracy"]
        ),
        "Precision": float(
            row["Precision"]
        ),
        "Recall": float(
            row["Recall"]
        ),
        "F1-score": float(
            row["F1-score"]
        ),
        "ROC-AUC": float(
            row["ROC-AUC"]
        ),
        "Threshold": float(
            row["Threshold"]
        ),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():

    if not V7_SCRIPT.exists():

        raise FileNotFoundError(
            f"V7 script not found: {V7_SCRIPT}"
        )

    results = []

    for seed in SEEDS:

        result = run_one_seed(
            seed
        )

        results.append(
            result
        )

        if result["Status"] == "SUCCESS":

            print(
                f"Seed {seed} completed:"
            )

            print(
                f"  Accuracy:  "
                f"{result['Accuracy']:.6f}"
            )

            print(
                f"  Precision: "
                f"{result['Precision']:.6f}"
            )

            print(
                f"  Recall:    "
                f"{result['Recall']:.6f}"
            )

            print(
                f"  F1-score:  "
                f"{result['F1-score']:.6f}"
            )

            print(
                f"  ROC-AUC:   "
                f"{result['ROC-AUC']:.6f}"
            )

        else:

            print(
                f"Seed {seed} failed:"
            )

            print(
                f"  {result.get('Error')}"
            )

    # ========================================================================
    # PER-SEED RESULTS
    # ========================================================================

    results_df = pd.DataFrame(
        results
    )

    per_seed_path = (
        OUTPUT_DIR
        / "v7_stability_per_seed.csv"
    )

    results_df.to_csv(
        per_seed_path,
        index=False,
    )

    # ========================================================================
    # SUCCESSFUL RUNS ONLY
    # ========================================================================

    successful = results_df[
        results_df["Status"]
        == "SUCCESS"
    ].copy()

    if successful.empty:

        raise RuntimeError(
            "All V7 stability runs failed."
        )

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
        "Threshold",
    ]

    # ========================================================================
    # MEAN / STD
    # ========================================================================

    summary_rows = []

    for metric in metric_columns:

        values = pd.to_numeric(
            successful[metric],
            errors="coerce",
        ).dropna()

        summary_rows.append(
            {
                "Metric":
                    metric,

                "Mean":
                    float(
                        values.mean()
                    ),

                "Std_Dev":
                    float(
                        values.std(
                            ddof=1
                        )
                    ),

                "Min":
                    float(
                        values.min()
                    ),

                "Max":
                    float(
                        values.max()
                    ),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        OUTPUT_DIR
        / "v7_stability_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # ========================================================================
    # JSON SUMMARY
    # ========================================================================

    json_summary = {
        "seeds": SEEDS,

        "successful_runs":
            int(
                len(successful)
            ),

        "failed_runs":
            int(
                len(results_df)
                - len(successful)
            ),

        "metrics": {
            row["Metric"]: {
                "mean":
                    row["Mean"],
                "std_dev":
                    row["Std_Dev"],
                "min":
                    row["Min"],
                "max":
                    row["Max"],
            }
            for _, row
            in summary_df.iterrows()
        },
    }

    json_path = (
        OUTPUT_DIR
        / "v7_stability_summary.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            json_summary,
            file,
            indent=2,
        )

    # ========================================================================
    # PRINT FINAL SUMMARY
    # ========================================================================

    print()
    print("=" * 75)
    print(
        "SH-DAS V7 STABILITY SUMMARY"
    )
    print("=" * 75)

    print()

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        f"Per-seed results:"
        f"\n{per_seed_path}"
    )

    print(
        f"\nMean/std summary:"
        f"\n{summary_path}"
    )

    print(
        f"\nJSON summary:"
        f"\n{json_path}"
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()