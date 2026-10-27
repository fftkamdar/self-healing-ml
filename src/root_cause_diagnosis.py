
from pathlib import Path

import pandas as pd

from self_healing_pipeline import (
    PSI_THRESHOLD,
    REPORTS_DIR,
    load_data,
    train_test_split,
    RANDOM_STATE,
    apply_synthetic_drift,
    calculate_feature_drift,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODERATE_PSI_THRESHOLD = 0.10


# ============================================================
# 2. FEATURE-LEVEL DIAGNOSIS
# ============================================================

def diagnose_numeric_feature(reference, current):
    """Describe how a numeric feature changed."""

    reference = pd.to_numeric(
        pd.Series(reference), errors="coerce"
    ).dropna()

    current = pd.to_numeric(
        pd.Series(current), errors="coerce"
    ).dropna()

    if reference.empty or current.empty:
        return {
            "reference_mean": None,
            "current_mean": None,
            "mean_change_percent": None,
            "reference_median": None,
            "current_median": None,
            "direction": "Insufficient numeric data",
            "diagnosis": "The feature could not be compared reliably.",
        }

    ref_mean = float(reference.mean())
    cur_mean = float(current.mean())
    ref_median = float(reference.median())
    cur_median = float(current.median())

    if abs(ref_mean) > 1e-12:
        mean_change_percent = (
            (cur_mean - ref_mean) / abs(ref_mean)
        ) * 100
    else:
        mean_change_percent = None

    if cur_mean > ref_mean:
        direction = "Upward shift"
    elif cur_mean < ref_mean:
        direction = "Downward shift"
    else:
        direction = "No mean change"

    if mean_change_percent is None:
        change_text = "The reference mean is zero, so percentage change is undefined."
    else:
        change_text = (
            f"The mean changed by {mean_change_percent:+.2f}%."
        )

    diagnosis = (
        f"{direction}. {change_text} "
        f"Median changed from {ref_median:.3f} "
        f"to {cur_median:.3f}."
    )

    return {
        "reference_mean": ref_mean,
        "current_mean": cur_mean,
        "mean_change_percent": mean_change_percent,
        "reference_median": ref_median,
        "current_median": cur_median,
        "direction": direction,
        "diagnosis": diagnosis,
    }


def diagnose_categorical_feature(reference, current):
    """Describe changes in the most common category."""

    reference = (
        pd.Series(reference)
        .fillna("__MISSING__")
        .astype(str)
    )

    current = (
        pd.Series(current)
        .fillna("__MISSING__")
        .astype(str)
    )

    ref_counts = reference.value_counts(normalize=True)
    cur_counts = current.value_counts(normalize=True)

    ref_top = str(ref_counts.index[0])
    cur_top = str(cur_counts.index[0])

    ref_share = float(ref_counts.iloc[0])
    cur_share = float(cur_counts.iloc[0])

    if ref_top == cur_top:
        direction = "Most common category unchanged"
        diagnosis = (
            f"The most common category remains '{ref_top}'. "
            f"Its share changed from {ref_share:.1%} "
            f"to {cur_share:.1%}."
        )
    else:
        direction = "Most common category changed"
        diagnosis = (
            f"The most common category changed from '{ref_top}' "
            f"to '{cur_top}'. Its share changed from "
            f"{ref_share:.1%} to {cur_share:.1%}."
        )

    return {
        "reference_mean": None,
        "current_mean": None,
        "mean_change_percent": None,
        "reference_median": None,
        "current_median": None,
        "direction": direction,
        "diagnosis": diagnosis,
    }


# ============================================================
# 3. RUN ROOT-CAUSE ANALYSIS
# ============================================================

def main():
    print("=" * 70)
    print("FEATURE-LEVEL ROOT-CAUSE DRIFT DIAGNOSIS")
    print("=" * 70)

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Use the test portion as a source for a simulated adaptation batch.
    X_adapt, _, _, _ = train_test_split(
        X_test,
        y_test,
        test_size=0.50,
        random_state=21,
        stratify=y_test,
    )

    X_adapt_shifted = apply_synthetic_drift(X_adapt)

    # Reuse the project's existing PSI implementation.
    psi_report = calculate_feature_drift(
        X_train,
        X_adapt_shifted,
    )

    rows = []

    for _, psi_row in psi_report.iterrows():
        feature = psi_row["feature"]

        reference = X_train[feature]
        current = X_adapt_shifted[feature]

        if pd.api.types.is_numeric_dtype(reference):
            details = diagnose_numeric_feature(reference, current)
        else:
            details = diagnose_categorical_feature(reference, current)

        psi = float(psi_row["psi"])

        if psi < MODERATE_PSI_THRESHOLD:
            severity = "Low"
        elif psi < PSI_THRESHOLD:
            severity = "Moderate"
        else:
            severity = "High"

        rows.append({
            "feature": feature,
            "psi": psi,
            "drift_level": severity,
            **details,
        })

    diagnosis_df = pd.DataFrame(rows).sort_values(
        "psi",
        ascending=False,
    )

    output_csv = REPORTS_DIR / "root_cause_diagnosis.csv"
    diagnosis_df.to_csv(output_csv, index=False)

    # --------------------------------------------------------
    # Overall interpretation
    # --------------------------------------------------------

    high_features = diagnosis_df.loc[
        diagnosis_df["drift_level"] == "High",
        "feature",
    ].tolist()

    moderate_features = diagnosis_df.loc[
        diagnosis_df["drift_level"] == "Moderate",
        "feature",
    ].tolist()

    total_features = len(diagnosis_df)
    affected_features = len(
        diagnosis_df[diagnosis_df["drift_level"] != "Low"]
    )

    if affected_features == 0:
        overall_diagnosis = (
            "No moderate or high feature-level drift was detected."
        )
    elif len(high_features) <= 2 and affected_features <= 2:
        overall_diagnosis = (
            "Drift is concentrated in a small number of features."
        )
    else:
        overall_diagnosis = (
            "Drift affects multiple features and may be distributed "
            "across the input data."
        )

    report_lines = [
        "FEATURE-LEVEL ROOT-CAUSE DRIFT DIAGNOSIS",
        "=" * 55,
        f"PSI threshold: {PSI_THRESHOLD}",
        f"Features analysed: {total_features}",
        f"Features with moderate or high drift: {affected_features}",
        f"High-drift features: {high_features}",
        f"Moderate-drift features: {moderate_features}",
        "",
        "OVERALL INTERPRETATION",
        overall_diagnosis,
        "",
        "FEATURE DETAILS",
        "-" * 55,
    ]

    for _, row in diagnosis_df.iterrows():
        report_lines.extend([
            f"Feature: {row['feature']}",
            f"PSI: {row['psi']:.6f}",
            f"Drift level: {row['drift_level']}",
            f"Direction: {row['direction']}",
            f"Diagnosis: {row['diagnosis']}",
            "",
        ])

    report_lines.extend([
        "IMPORTANT LIMITATION",
        (
            "PSI and distribution summaries identify changes in input "
            "data. They do not, by themselves, prove that model "
            "performance has degraded or establish a causal root cause."
        ),
        (
            "This experiment uses synthetically shifted data, so its "
            "findings should not be interpreted as evidence of live "
            "production drift."
        ),
    ])

    output_txt = REPORTS_DIR / "root_cause_diagnosis.txt"
    output_txt.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\nOverall diagnosis:")
    print(overall_diagnosis)

    print("\nHigh-drift features:")
    print(high_features if high_features else "None")

    print("\nFeature diagnosis table:")
    print(
        diagnosis_df[
            ["feature", "psi", "drift_level", "direction"]
        ].to_string(index=False)
    )

    print("\nSaved reports:")
    print(output_csv)
    print(output_txt)


if __name__ == "__main__":
    main()