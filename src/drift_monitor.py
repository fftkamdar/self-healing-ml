
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# SELF-HEALING ML
# Feature-Level Data Drift Monitoring using PSI
# ============================================================

DATA_PATH = Path("data/raw/Telco-Customer-Churn.csv")
REPORT_PATH = Path("reports/drift_report.txt")
FEATURE_REPORT_PATH = Path("reports/feature_drift_scores.csv")

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

EPSILON = 1e-6


# ============================================================
# 1. LOAD AND PREPARE DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df = df.dropna(subset=["Churn"])

X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].astype(int)

# Reproduce the split used by the baseline experiment.
reference, current = train_test_split(
    X,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

reference = reference.copy()
current = current.copy()

print("Reference dataset rows:", len(reference))
print("Current dataset rows:", len(current))


# ============================================================
# 2. SIMULATE CONTROLLED DATA DRIFT
# ============================================================

print("\nApplying controlled drift simulation...")

# Deliberately shift selected features in the current data.
# This is a simulation, not evidence of real-world drift.

current["tenure"] = (
    current["tenure"] * 1.5
).clip(upper=72)

current["MonthlyCharges"] = (
    current["MonthlyCharges"] * 1.2
).clip(upper=120)

print("Drift simulation applied to:")
print("- tenure")
print("- MonthlyCharges")


# ============================================================
# 3. PSI CALCULATION
# ============================================================

def calculate_psi(reference_proportions, current_proportions):
    """
    Calculate Population Stability Index (PSI).

    Both inputs must contain proportions for matching buckets.
    A small epsilon prevents logarithms of zero.
    """

    reference_proportions = np.asarray(
        reference_proportions, dtype=float
    )

    current_proportions = np.asarray(
        current_proportions, dtype=float
    )

    reference_proportions = np.clip(
        reference_proportions, EPSILON, None
    )

    current_proportions = np.clip(
        current_proportions, EPSILON, None
    )

    reference_proportions = (
        reference_proportions / reference_proportions.sum()
    )

    current_proportions = (
        current_proportions / current_proportions.sum()
    )

    psi = np.sum(
        (current_proportions - reference_proportions)
        * np.log(current_proportions / reference_proportions)
    )

    return float(psi)


# ============================================================
# 4. NUMERIC FEATURE PSI
# ============================================================

def numeric_psi(reference_series, current_series):
    """
    Calculate PSI for a numeric feature.

    Bins are determined from reference-data quantiles.
    Missing values are handled as a separate bucket.
    """

    ref_all = pd.to_numeric(
        reference_series, errors="coerce"
    )

    cur_all = pd.to_numeric(
        current_series, errors="coerce"
    )

    ref = ref_all.dropna()
    cur = cur_all.dropna()

    if ref.empty or cur.empty:
        return np.nan

    # Create quantile-based bin boundaries using reference data.
    quantiles = np.linspace(0, 1, 11)
    boundaries = np.unique(
        ref.quantile(quantiles).to_numpy()
    )

    if len(boundaries) < 2:
        # A constant reference feature has only one value bucket.
        ref_counts = np.array([len(ref)], dtype=float)
        cur_counts = np.array(
            [(cur == ref.iloc[0]).sum()], dtype=float
        )

        # Values different from the reference constant are grouped
        # into a separate bucket.
        different_count = len(cur) - cur_counts[0]

        ref_counts = np.append(ref_counts, 0.0)
        cur_counts = np.append(cur_counts, different_count)

    else:
        edges = np.concatenate(
            ([-np.inf], boundaries[1:-1], [np.inf])
        )

        edges = np.unique(edges)

        ref_bins = pd.cut(
            ref, bins=edges, include_lowest=True
        )

        cur_bins = pd.cut(
            cur, bins=edges, include_lowest=True
        )

        categories = ref_bins.cat.categories

        ref_counts = (
            ref_bins.value_counts(sort=False)
            .reindex(categories, fill_value=0)
            .to_numpy(dtype=float)
        )

        cur_counts = (
            cur_bins.value_counts(sort=False)
            .reindex(categories, fill_value=0)
            .to_numpy(dtype=float)
        )

    # Include missing values as a separate bucket.
    ref_missing = float(ref_all.isna().sum())
    cur_missing = float(cur_all.isna().sum())

    ref_counts = np.append(ref_counts, ref_missing)
    cur_counts = np.append(cur_counts, cur_missing)

    ref_props = ref_counts / max(len(ref_all), 1)
    cur_props = cur_counts / max(len(cur_all), 1)

    return calculate_psi(ref_props, cur_props)


# ============================================================
# 5. CATEGORICAL FEATURE PSI
# ============================================================

def categorical_psi(reference_series, current_series):
    """
    Calculate PSI using category frequencies.
    Categories appearing in either dataset are included.
    Missing values are treated as a category.
    """

    ref = reference_series.fillna("MISSING").astype(str)
    cur = current_series.fillna("MISSING").astype(str)

    categories = sorted(
        set(ref.unique()) | set(cur.unique())
    )

    ref_counts = (
        ref.value_counts()
        .reindex(categories, fill_value=0)
        .to_numpy(dtype=float)
    )

    cur_counts = (
        cur.value_counts()
        .reindex(categories, fill_value=0)
        .to_numpy(dtype=float)
    )

    ref_props = ref_counts / max(len(ref), 1)
    cur_props = cur_counts / max(len(cur), 1)

    return calculate_psi(ref_props, cur_props)


# ============================================================
# 6. MONITOR EVERY FEATURE
# ============================================================

print("\nCalculating feature-level PSI scores...")

results = []

for column in reference.columns:

    if pd.api.types.is_numeric_dtype(reference[column]):
        score = numeric_psi(
            reference[column], current[column]
        )
        method = "Numeric quantile bins"

    else:
        score = categorical_psi(
            reference[column], current[column]
        )
        method = "Categorical frequency"

    if pd.isna(score):
        status = "Could not calculate"
    elif score < 0.10:
        status = "Low drift"
    elif score < 0.25:
        status = "Moderate drift"
    else:
        status = "High drift"

    results.append({
        "feature": column,
        "psi": score,
        "status": status,
        "method": method,
    })


results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="psi",
    ascending=False,
    na_position="last",
)


# ============================================================
# 7. OVERALL DRIFT INDICATOR
# ============================================================

valid_scores = results_df["psi"].dropna()

if valid_scores.empty:

    maximum_psi = np.nan
    overall_status = "Unable to determine"

else:

    maximum_psi = float(valid_scores.max())

    if maximum_psi < 0.10:
        overall_status = "Low drift"

    elif maximum_psi < 0.25:
        overall_status = "Moderate drift"

    else:
        overall_status = "High drift"


# ============================================================
# 8. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 65)
print("SELF-HEALING ML: FEATURE-LEVEL DRIFT RESULTS")
print("=" * 65)

print(results_df.to_string(index=False))

print("\nMaximum feature PSI:", maximum_psi)
print("Overall drift indicator:", overall_status)


# ============================================================
# 9. SAVE REPORTS
# ============================================================

with open(REPORT_PATH, "w", encoding="utf-8") as file:

    file.write("SELF-HEALING ML - DRIFT MONITOR REPORT\n")
    file.write("=" * 50 + "\n\n")

    file.write("Experiment: Controlled simulated feature drift\n")
    file.write("Dataset: IBM Telco Customer Churn sample\n")
    file.write("Reference: 80% training split\n")
    file.write("Current: 20% test split with selected shifts\n")
    file.write("Modified features: tenure, MonthlyCharges\n\n")

    file.write("FEATURE-LEVEL RESULTS\n")
    file.write(results_df.to_string(index=False))
    file.write("\n\n")

    file.write(f"Maximum feature PSI: {maximum_psi}\n")
    file.write(f"Overall drift indicator: {overall_status}\n\n")

    file.write(
        "Interpretation:\n"
        "PSI measures changes in feature distributions. "
        "It does not directly measure model predictive performance. "
        "The current dataset was deliberately modified for this "
        "controlled experiment. Drift thresholds are heuristic "
        "guidelines and should be validated for the application.\n"
    )

results_df.to_csv(
    FEATURE_REPORT_PATH, index=False
)

print("\nReports saved successfully:")
print(REPORT_PATH)
print(FEATURE_REPORT_PATH)

print("\nDrift-monitoring experiment completed.")