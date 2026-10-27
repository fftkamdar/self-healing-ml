from pathlib import Path
import json
import math

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

VERSIONS = [
    ("V8", REPORTS / "proposed_model_v8" / "v8_summary.json"),
    ("V9", REPORTS / "proposed_model_v9" / "v9_summary.json"),
    ("V10", REPORTS / "proposed_model_v10" / "v10_summary.json"),
    ("V11", REPORTS / "proposed_model_v11" / "v11_summary.json"),
    ("V12", REPORTS / "proposed_model_v12" / "v12_summary.json"),
]

METRICS = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score",
    "ROC-AUC",
]


def load_json(path):
    if not path.exists():
        return None

    try:
        return json.loads(
            path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        print(f"Could not read {path}: {exc}")
        return None


def get_test_metrics(data):
    if not data:
        return None

    candidates = [
        data.get("test_metrics"),
        data.get("final_test_metrics"),
        data.get("metrics"),
    ]

    for item in candidates:
        if isinstance(item, dict):
            if any(metric in item for metric in METRICS):
                return item

    return None


def get_reference_metrics(data):
    if not data:
        return None

    candidates = [
        data.get("reference_metrics"),
        data.get("existing_reference"),
        data.get("existing_best"),
    ]

    for item in candidates:
        if isinstance(item, dict):
            if any(metric in item for metric in METRICS):
                return item

    return None


def fmt(value):
    if value is None:
        return "N/A"

    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def main():
    print("=" * 100)
    print("SELF-HEALING ML — EXPERIMENT AUDIT")
    print("=" * 100)

    history = []

    for version, path in VERSIONS:
        data = load_json(path)

        if data is None:
            print(f"\n{version}: summary not found")
            continue

        test_metrics = get_test_metrics(data)

        print(f"\n{version}")
        print("-" * 100)
        print(f"File: {path}")

        if "selected_model" in data:
            print(
                f"Selected model: "
                f"{data.get('selected_model')}"
            )

        if "selected_approach" in data:
            print(
                f"Selected approach: "
                f"{data.get('selected_approach')}"
            )

        if "selected_threshold" in data:
            print(
                f"Threshold: "
                f"{data.get('selected_threshold')}"
            )

        if "selected_model" not in data and "model" in data:
            print(
                f"Model: {data.get('model')}"
            )

        if test_metrics:
            print("Test metrics:")
            for metric in METRICS:
                print(
                    f"  {metric:<12}: "
                    f"{fmt(test_metrics.get(metric))}"
                )

            history.append(
                {
                    "version": version,
                    "metrics": test_metrics,
                    "data": data,
                }
            )
        else:
            print("Test metrics: NOT FOUND")

        if "dominance_count" in data:
            print(
                f"Dominance: "
                f"{data.get('dominance_count')}/5"
            )

        if "beats_all_five" in data:
            print(
                "Beats all five: "
                f"{data.get('beats_all_five')}"
            )

        if "v11_vs_v9_dominance" in data:
            print(
                f"V11 vs V9 dominance: "
                f"{data.get('v11_vs_v9_dominance')}/5"
            )

        if "v12_vs_v9_dominance" in data:
            print(
                f"V12 vs V9 dominance: "
                f"{data.get('v12_vs_v9_dominance')}/5"
            )

    print("\n" + "=" * 100)
    print("PROPOSED-MODEL HISTORY")
    print("=" * 100)

    if not history:
        print("No experiment summaries were found.")
        return

    header = (
        f"{'Version':<10}"
        f"{'Accuracy':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'ROC-AUC':>12}"
    )

    print(header)
    print("-" * len(header))

    for item in history:
        m = item["metrics"]

        print(
            f"{item['version']:<10}"
            f"{fmt(m.get('Accuracy')):>12}"
            f"{fmt(m.get('Precision')):>12}"
            f"{fmt(m.get('Recall')):>12}"
            f"{fmt(m.get('F1-score')):>12}"
            f"{fmt(m.get('ROC-AUC')):>12}"
        )

    # Find strongest version for each metric.
    print("\n" + "=" * 100)
    print("METRIC-BY-METRIC HISTORY")
    print("=" * 100)

    for metric in METRICS:
        available = []

        for item in history:
            value = item["metrics"].get(metric)

            if isinstance(value, (int, float)):
                if math.isfinite(float(value)):
                    available.append(
                        (
                            item["version"],
                            float(value),
                        )
                    )

        if not available:
            continue

        best_version, best_value = max(
            available,
            key=lambda x: x[1],
        )

        print(
            f"{metric:<12}: "
            f"{best_version} = {best_value:.4f}"
        )

    # Compare V9 against later versions.
    v9 = next(
        (
            item
            for item in history
            if item["version"] == "V9"
        ),
        None,
    )

    if v9:
        print("\n" + "=" * 100)
        print("V9 AS CURRENT PROPOSED BENCHMARK")
        print("=" * 100)

        for item in history:
            if item["version"] == "V9":
                continue

            print(f"\n{item['version']} vs V9")

            improved = 0
            worsened = 0
            same = 0

            for metric in METRICS:
                a = item["metrics"].get(metric)
                b = v9["metrics"].get(metric)

                if not isinstance(a, (int, float)):
                    continue

                if not isinstance(b, (int, float)):
                    continue

                delta = float(a) - float(b)

                if delta > 0:
                    status = "IMPROVED"
                    improved += 1
                elif delta < 0:
                    status = "WORSE"
                    worsened += 1
                else:
                    status = "SAME"
                    same += 1

                print(
                    f"  {metric:<12}: "
                    f"{delta:+.4f} "
                    f"{status}"
                )

            print(
                f"  Summary: "
                f"{improved} improved, "
                f"{worsened} worse, "
                f"{same} same"
            )

    # Collect reference metrics to detect inconsistencies.
    print("\n" + "=" * 100)
    print("REFERENCE-METRIC CONSISTENCY CHECK")
    print("=" * 100)

    reference_sets = []

    for version, path in VERSIONS:
        data = load_json(path)
        ref = get_reference_metrics(data)

        if ref:
            reference_sets.append(
                (
                    version,
                    {
                        metric: ref.get(metric)
                        for metric in METRICS
                    },
                )
            )

    if not reference_sets:
        print(
            "No reference-metric blocks were found "
            "in the saved summaries."
        )
    else:
        for version, ref in reference_sets:
            print(f"\n{version} reference:")
            for metric in METRICS:
                print(
                    f"  {metric:<12}: "
                    f"{fmt(ref.get(metric))}"
                )

        print(
            "\nNOTE: Different reference values do not "
            "automatically mean an experiment is wrong."
        )
        print(
            "They can indicate that different benchmark "
            "setups were used."
        )

    print("\n" + "=" * 100)
    print("AUDIT CONCLUSION")
    print("=" * 100)

    print(
        "1. No model was retrained by this audit."
    )
    print(
        "2. Saved V8-V12 experiment results were "
        "read from their JSON summaries."
    )
    print(
        "3. V9 remains a candidate benchmark unless "
        "the audit reveals a stronger later result."
    )
    print(
        "4. We will not create another model version "
        "until the benchmark setup is consistent."
    )

    output_path = (
        REPORTS
        / "experiment_audit"
        / "experiment_audit_summary.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    export = {
        "versions_found": [
            item["version"]
            for item in history
        ],
        "history": [
            {
                "version": item["version"],
                "metrics": item["metrics"],
            }
            for item in history
        ],
        "reference_sets": [
            {
                "version": version,
                "reference": reference,
            }
            for version, reference in reference_sets
        ],
    }

    output_path.write_text(
        json.dumps(
            export,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"\nSaved audit summary:\n{output_path}"
    )


if __name__ == "__main__":
    main()
