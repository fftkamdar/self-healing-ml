from pathlib import Path
import sys
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))

from drift_relevance_alignment import compute_dra
from selective_retraining_experiment import (
    compute_feature_psi,
    compute_feature_contribution,
    HIGH_DRIFT_PSI_THRESHOLD,
    DATA_PATH,
    BASELINE_MODEL_PATH,
)


def clean_telco_data(df):

    df = df.copy()

    # Remove ID column
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Convert TotalCharges to numeric.
    # Blank strings in the original IBM Telco dataset
    # become NaN and are handled by the model's imputer.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce",
        )

    # Convert target
    if "Churn" in df.columns:
        df["Churn"] = (
            df["Churn"]
            .map({
                "Yes": 1,
                "No": 0,
            })
        )

    return df


def main():

    print("=" * 78)
    print("SH-DAS - DRIFT-RELEVANCE ALIGNMENT (DRA) EXPERIMENT")
    print("=" * 78)

    # ---------------------------------------------------------
    # Load and clean Telco dataset
    # ---------------------------------------------------------
    df = pd.read_csv(DATA_PATH)

    df = clean_telco_data(df)

    X = df.drop(
        columns=["Churn"],
        errors="ignore",
    )

    y = (
        df["Churn"]
        if "Churn" in df.columns
        else None
    )

    # ---------------------------------------------------------
    # Reference/current split
    # ---------------------------------------------------------
    split = int(len(X) * 0.80)

    X_reference = X.iloc[:split].copy()
    X_current = X.iloc[split:].copy()

    print()
    print("Reference rows:", len(X_reference))
    print("Current rows:", len(X_current))

    # ---------------------------------------------------------
    # Feature contribution
    # ---------------------------------------------------------
    print()
    print("Computing feature contribution...")

    contribution = compute_feature_contribution(
        joblib.load(BASELINE_MODEL_PATH),
        X_reference,
        y.iloc[:split] if y is not None else None,
    )

    # ---------------------------------------------------------
    # Controlled drift scenarios
    # ---------------------------------------------------------
    scenarios = {
        "No drift": {},

        "Mild tenure drift": {
            "tenure": 1.10,
        },

        "Moderate tenure drift": {
            "tenure": 1.30,
        },

        "Strong tenure drift": {
            "tenure": 1.50,
        },

        "Strong MonthlyCharges drift": {
            "MonthlyCharges": 1.20,
        },

        "Strong important-feature drift": {
            "tenure": 1.50,
            "MonthlyCharges": 1.20,
        },
    }

    results = []
    detail_tables = []

    # ---------------------------------------------------------
    # Calculate DRA
    # ---------------------------------------------------------
    for scenario_name, multipliers in scenarios.items():

        current = X_current.copy()

        for feature, multiplier in multipliers.items():

            if feature in current.columns:

                current[feature] = (
                    current[feature]
                    * multiplier
                )

        drift_table = compute_feature_psi(
            X_reference,
            current,
        )

        dra_score, dra_details = compute_dra(
            drift_table,
            contribution,
            high_drift_threshold=HIGH_DRIFT_PSI_THRESHOLD,
        )

        dra_trigger = dra_score >= 0.50

        print()
        print("-" * 78)
        print(scenario_name)
        print("-" * 78)

        print(
            "Maximum PSI: "
            f"{drift_table['PSI'].max():.4f}"
        )

        print(
            "DRA Score: "
            f"{dra_score:.4f}"
        )

        print(
            "DRA Trigger (threshold 0.50): "
            f"{dra_trigger}"
        )

        top_features = (
            dra_details[
                dra_details["Alignment_Contribution"] > 0
            ]
            .head(5)["Feature"]
            .tolist()
        )

        print(
            "Top drift-relevance features:",
            top_features,
        )

        details = dra_details.copy()

        details["Scenario"] = scenario_name
        details["DRA_Score"] = dra_score
        details["DRA_Trigger"] = dra_trigger

        detail_tables.append(details)

        results.append(
            {
                "Scenario": scenario_name,
                "Maximum_PSI": (
                    drift_table["PSI"].max()
                ),
                "DRA_Score": dra_score,
                "DRA_Trigger": dra_trigger,
                "Top_Aligned_Feature": (
                    top_features[0]
                    if top_features
                    else ""
                ),
            }
        )

    # ---------------------------------------------------------
    # Save evidence
    # ---------------------------------------------------------
    output_dir = (
        ROOT
        / "reports"
        / "selective_retraining"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        output_dir
        / "dra_experiment_summary.csv"
    )

    details_path = (
        output_dir
        / "dra_feature_alignment.csv"
    )

    pd.DataFrame(results).to_csv(
        summary_path,
        index=False,
    )

    pd.concat(
        detail_tables,
        ignore_index=True,
    ).to_csv(
        details_path,
        index=False,
    )

    print()
    print("=" * 78)
    print("DRA EXPERIMENT COMPLETED")
    print("=" * 78)
    print("Summary:")
    print(summary_path)
    print("Feature-level evidence:")
    print(details_path)


if __name__ == "__main__":
    main()
