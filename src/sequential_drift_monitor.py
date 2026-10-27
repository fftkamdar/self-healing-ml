
"""
Sequential Drift Monitor
------------------------
Simulates incoming batches and monitors feature distribution drift.

The batch creation method preserves the original DataFrame column
types so numeric features are not mistakenly treated as categorical.
"""

import numpy as np
import pandas as pd

from self_healing_pipeline import (
    REPORTS_DIR,
    load_data,
    train_test_split,
    RANDOM_STATE,
    apply_synthetic_drift,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

BATCH_COUNT = 6
PERSISTENCE_REQUIRED = 3
PSI_THRESHOLD = 0.25


# ============================================================
# 2. BATCH CREATION
# ============================================================

def create_batches(data, batch_count):
    """Split a DataFrame while preserving its column data types."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if batch_count < 1:
        raise ValueError("batch_count must be at least 1.")

    if data.empty:
        raise ValueError("Cannot split an empty DataFrame.")

    positions = np.array_split(np.arange(len(data)), batch_count)

    # Use iloc on the original DataFrame to preserve column dtypes.
    return [
        data.iloc[indexes].copy().reset_index(drop=True)
        for indexes in positions
        if len(indexes) > 0
    ]


def apply_batch_scenario(batch, batch_number):
    """
    Batches 1-2: no intentional shift.
    Batch 3: mild shift.
    Batches 4-6: stronger synthetic shift.
    """

    shifted = batch.copy()

    if batch_number <= 2:
        return shifted

    if batch_number == 3:
        if "tenure" in shifted.columns:
            shifted["tenure"] = (
                pd.to_numeric(shifted["tenure"], errors="coerce") * 1.10
            ).clip(upper=72)

        if "MonthlyCharges" in shifted.columns:
            shifted["MonthlyCharges"] = (
                pd.to_numeric(
                    shifted["MonthlyCharges"], errors="coerce"
                ) * 1.05
            ).clip(upper=120)

        return shifted

    return apply_synthetic_drift(shifted)


# ============================================================
# 3. PSI FUNCTIONS
# ============================================================

def calculate_numeric_psi(reference, current, bins=10):
    """Calculate numeric PSI using quantile bins from reference data."""

    reference = pd.to_numeric(
        pd.Series(reference), errors="coerce"
    ).dropna()

    current = pd.to_numeric(
        pd.Series(current), errors="coerce"
    ).dropna()

    if reference.empty or current.empty:
        return 0.0

    edges = np.unique(
        reference.quantile(np.linspace(0, 1, bins + 1)).to_numpy()
    )

    if len(edges) < 2:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    epsilon = 1e-6

    ref_pct = ref_counts.astype(float) / max(ref_counts.sum(), 1)
    cur_pct = cur_counts.astype(float) / max(cur_counts.sum(), 1)

    ref_pct = np.clip(ref_pct, epsilon, None)
    cur_pct = np.clip(cur_pct, epsilon, None)

    ref_pct /= ref_pct.sum()
    cur_pct /= cur_pct.sum()

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def calculate_categorical_psi(reference, current):
    """Calculate categorical PSI using the union of observed categories."""

    reference = pd.Series(reference).fillna("__MISSING__").astype(str)
    current = pd.Series(current).fillna("__MISSING__").astype(str)

    if reference.empty or current.empty:
        return 0.0

    categories = sorted(set(reference.unique()) | set(current.unique()))

    ref_counts = reference.value_counts().reindex(
        categories, fill_value=0
    ).to_numpy(dtype=float)

    cur_counts = current.value_counts().reindex(
        categories, fill_value=0
    ).to_numpy(dtype=float)

    epsilon = 1e-6

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    ref_pct = np.clip(ref_pct, epsilon, None)
    cur_pct = np.clip(cur_pct, epsilon, None)

    ref_pct /= ref_pct.sum()
    cur_pct /= cur_pct.sum()

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def calculate_feature_psi(reference_df, current_df):
    """Calculate PSI for every feature shared by the two DataFrames."""

    records = []

    shared_columns = [
        column for column in reference_df.columns
        if column in current_df.columns
    ]

    for column in shared_columns:
        reference_series = reference_df[column]
        current_series = current_df[column]

        # Determine numeric status from the reference schema and
        # coerce values defensively in case the current data differs.
        reference_numeric = pd.to_numeric(
            reference_series, errors="coerce"
        )
        current_numeric = pd.to_numeric(
            current_series, errors="coerce"
        )

        reference_nonmissing = reference_series.notna().sum()
        numeric_reference_count = reference_numeric.notna().sum()

        is_numeric = (
            pd.api.types.is_numeric_dtype(reference_series)
            or (
                reference_nonmissing > 0
                and numeric_reference_count == reference_nonmissing
            )
        )

        if is_numeric:
            psi = calculate_numeric_psi(
                reference_numeric,
                current_numeric,
            )
            feature_type = "numeric"
        else:
            psi = calculate_categorical_psi(
                reference_series,
                current_series,
            )
            feature_type = "categorical"

        if psi < 0.10:
            drift_level = "LOW"
        elif psi < PSI_THRESHOLD:
            drift_level = "MODERATE"
        else:
            drift_level = "HIGH"

        records.append({
            "feature": column,
            "feature_type": feature_type,
            "psi": psi,
            "drift_level": drift_level,
        })

    return pd.DataFrame(records)


# ============================================================
# 4. SEQUENTIAL MONITORING
# ============================================================

def main():
    print("=" * 70)
    print("SEQUENTIAL DRIFT MONITOR")
    print("=" * 70)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    batches = create_batches(X_test, BATCH_COUNT)

    print(f"Reference rows: {len(X_train)}")
    print(f"Monitoring rows: {len(X_test)}")
    print(f"Number of batches: {len(batches)}")
    print(f"PSI high-drift threshold: {PSI_THRESHOLD}")
    print(
        "Persistent alert requirement: "
        f"{PERSISTENCE_REQUIRED} consecutive drift batches"
    )

    summary_records = []
    feature_records = []

    consecutive_drift_batches = 0
    alert_batch = None
    persistent_alert_triggered = False

    for batch_number, original_batch in enumerate(batches, start=1):
        current_batch = apply_batch_scenario(
            original_batch,
            batch_number,
        )

        feature_report = calculate_feature_psi(
            X_train,
            current_batch,
        )

        if feature_report.empty:
            max_psi = 0.0
            high_features = []
        else:
            max_psi = float(feature_report["psi"].max())
            high_features = feature_report.loc[
                feature_report["psi"] >= PSI_THRESHOLD,
                "feature",
            ].tolist()

        drift_detected = max_psi >= PSI_THRESHOLD

        if drift_detected:
            consecutive_drift_batches += 1
        else:
            consecutive_drift_batches = 0

        if (
            consecutive_drift_batches >= PERSISTENCE_REQUIRED
            and not persistent_alert_triggered
        ):
            persistent_alert_triggered = True
            alert_batch = batch_number

        if persistent_alert_triggered:
            status = "PERSISTENT_DRIFT_ALERT"
        elif drift_detected:
            status = "DRIFT_OBSERVED"
        else:
            status = "NORMAL"

        summary_records.append({
            "batch_number": batch_number,
            "row_count": len(current_batch),
            "max_psi": max_psi,
            "drift_detected": drift_detected,
            "consecutive_drift_batches": consecutive_drift_batches,
            "high_drift_features": ", ".join(high_features),
            "status": status,
        })

        feature_report["batch_number"] = batch_number
        feature_report["row_count"] = len(current_batch)
        feature_records.append(feature_report)

        print(
            f"Batch {batch_number}: "
            f"rows={len(current_batch)}, "
            f"max PSI={max_psi:.6f}, "
            f"drift={'YES' if drift_detected else 'NO'}, "
            f"consecutive={consecutive_drift_batches}, "
            f"status={status}"
        )

        if high_features:
            print("  High-drift features:", ", ".join(high_features))

    # --------------------------------------------------------
    # Save reports
    # --------------------------------------------------------

    summary_df = pd.DataFrame(summary_records)
    feature_df = (
        pd.concat(feature_records, ignore_index=True)
        if feature_records
        else pd.DataFrame()
    )

    summary_path = REPORTS_DIR / "sequential_drift_summary.csv"
    feature_path = REPORTS_DIR / "sequential_feature_drift.csv"

    summary_df.to_csv(summary_path, index=False)
    feature_df.to_csv(feature_path, index=False)

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SEQUENTIAL MONITORING SUMMARY")
    print("=" * 70)

    print(
        summary_df[
            [
                "batch_number",
                "max_psi",
                "drift_detected",
                "consecutive_drift_batches",
                "status",
            ]
        ].to_string(index=False)
    )

    if persistent_alert_triggered:
        print(f"\nPersistent drift alert triggered at batch {alert_batch}.")
    else:
        print("\nNo persistent drift alert was triggered.")

    print("\nInterpretation:")
    print(
        "The first two batches have no intentional synthetic shift, "
        "but may still show sampling variation relative to the reference."
    )
    print(
        "The alert indicates feature-distribution change under the "
        "chosen PSI threshold. It does not alone prove model degradation "
        "or justify automatic retraining."
    )
    print(
        "This is a synthetic experiment, not a live production monitor."
    )

    print("\nSaved reports:")
    print(summary_path)
    print(feature_path)


if __name__ == "__main__":
    main()