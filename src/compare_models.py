
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = PROJECT_ROOT / "reports" / "baseline_metrics.txt"
OPTUNA_PATH = PROJECT_ROOT / "reports" / "optuna_tuning_report.txt"

CSV_PATH = PROJECT_ROOT / "reports" / "model_comparison.csv"
CHART_PATH = PROJECT_ROOT / "reports" / "model_comparison.png"



def read_metric(report_text, metric_name):
    """Extract the final occurrence of a metric from a text report."""
    pattern = rf"{re.escape(metric_name)}:\s*([0-9]*\.?[0-9]+)"
    matches = re.findall(pattern, report_text, flags=re.IGNORECASE)

    if not matches:
        raise ValueError(f"Could not find '{metric_name}' in report.")

    # The Optuna report lists validation ROC-AUC before test ROC-AUC.
    # Taking the final occurrence selects the held-out test result.
    return float(matches[-1])


def main():
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(f"Missing report: {BASELINE_PATH}")

    if not OPTUNA_PATH.exists():
        raise FileNotFoundError(f"Missing report: {OPTUNA_PATH}")

    baseline_text = BASELINE_PATH.read_text(encoding="utf-8")
    optuna_text = OPTUNA_PATH.read_text(encoding="utf-8")

    metrics = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]

    rows = []
    for metric in metrics:
        rows.append(
            {
                "Metric": metric,
                "Baseline": read_metric(baseline_text, metric),
                "Optuna_Tuned": read_metric(optuna_text, metric),
            }
        )

    comparison = pd.DataFrame(rows)
    comparison["Change"] = (
        comparison["Optuna_Tuned"] - comparison["Baseline"]
    )

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(CSV_PATH, index=False)

    print("=" * 65)
    print("BASELINE VS OPTUNA MODEL COMPARISON")
    print("=" * 65)
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Create a grouped bar chart.
    x_positions = range(len(comparison))
    bar_width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(
        [x - bar_width / 2 for x in x_positions],
        comparison["Baseline"],
        width=bar_width,
        label="Baseline",
    )
    plt.bar(
        [x + bar_width / 2 for x in x_positions],
        comparison["Optuna_Tuned"],
        width=bar_width,
        label="Optuna Tuned",
    )

    plt.xticks(list(x_positions), comparison["Metric"])
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Baseline vs Optuna-Tuned Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=300)
    plt.close()

    print(f"\nCSV saved: {CSV_PATH}")
    print(f"Chart saved: {CHART_PATH}")
    print("\nComparison report created successfully.")


if __name__ == "__main__":
    main()