"""
SELECTIVE RETRAINING & COST ANALYSIS
Self-Healing ML project

Research question addressed:
"Can feature-level drift diagnosis reduce unnecessary retraining while
maintaining or improving predictive performance?"

Compares two policies across controlled drift scenarios:
1) Full retraining policy: retrain whenever max PSI is high.
2) Feature-aware selective policy: retrain only when high drift overlaps
   with meaningful measured feature contribution.

This is an offline controlled experiment. Saved models are never overwritten.
Run from the project root:
    python src/selective_retraining_experiment.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
BASELINE_MODEL_PATH = PROJECT_ROOT / "models" / "baseline_churn_pipeline.joblib"
REPORT_DIR = PROJECT_ROOT / "reports" / "selective_retraining"
SCENARIO_REPORT_PATH = REPORT_DIR / "selective_retraining_scenarios.csv"
SUMMARY_REPORT_PATH = REPORT_DIR / "selective_retraining_summary.csv"
FEATURE_REPORT_PATH = REPORT_DIR / "selective_retraining_feature_contribution.csv"
TEXT_REPORT_PATH = REPORT_DIR / "selective_retraining_report.txt"
CONFIG_PATH = REPORT_DIR / "selective_retraining_config.json"

RANDOM_STATE = 42
HIGH_DRIFT_PSI_THRESHOLD = 0.25 #Prevents unnecessary retraining by requiring both high feature drift (PSI) and high feature contribution (ROC-AUC importance)
CONTRIBUTION_THRESHOLD = 0.03 #dual condition strategy


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    if "Churn" not in df.columns:
        raise ValueError("Expected target column 'Churn' was not found.")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["customerID", "Churn"], errors="ignore").copy()


def binary_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")
        if values.isna().any():
            raise ValueError("Target contains non-numeric values that could not be converted.")
        return values.astype(int)

    normalized = series.astype(str).str.strip().str.lower()
    mapping = {
        "yes": 1,
        "y": 1,
        "true": 1,
        "1": 1,
        "no": 0,
        "n": 0,
        "false": 0,
        "0": 0,
    }
    mapped = normalized.map(mapping)
    if mapped.isna().any():
        unique_values = normalized.dropna().unique().tolist()
        if len(unique_values) == 2:
            mapped = normalized.map({unique_values[0]: 0, unique_values[1]: 1})
    if mapped.isna().any():
        raise ValueError("Could not safely convert Churn to binary values.")
    return mapped.astype(int)


def psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref = pd.to_numeric(reference, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    cur = pd.to_numeric(current, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if ref.empty or cur.empty or ref.nunique() <= 1:
        return 0.0
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(ref.quantile(quantiles).to_numpy())
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_bins = pd.cut(ref, bins=edges, include_lowest=True)
    cur_bins = pd.cut(cur, bins=edges, include_lowest=True)
    ref_pct = ref_bins.value_counts(sort=False, normalize=True).to_numpy(dtype=float)
    cur_pct = cur_bins.value_counts(sort=False, normalize=True).to_numpy(dtype=float)
    eps = 1e-6
    ref_pct = np.where(ref_pct <= 0, eps, ref_pct)
    cur_pct = np.where(cur_pct <= 0, eps, cur_pct)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_feature_psi(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    numeric_features = [
        c for c in reference.select_dtypes(include="number").columns if c in current.columns
    ]
    rows = []
    for feature in numeric_features:
        rows.append({"Feature": feature, "PSI": psi_numeric(reference[feature], current[feature])})
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values("PSI", ascending=False).reset_index(drop=True)


def apply_multiplier(series: pd.Series, multiplier: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    finite = values.replace([np.inf, -np.inf], np.nan)
    upper = finite.quantile(0.995)
    shifted = values * multiplier
    if pd.notna(upper):
        shifted = shifted.clip(upper=upper * 1.15)
    return shifted


def simulate_scenario(current: pd.DataFrame, scenario: dict) -> pd.DataFrame:
    drifted = current.copy()
    for feature, multiplier in scenario.get("multipliers", {}).items():
        if feature in drifted.columns:
            drifted[feature] = apply_multiplier(drifted[feature], float(multiplier))
    return drifted


def find_high_drift_multiplier(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature: str,
    threshold: float,
    minimum: float = 1.05,
    maximum: float = 3.00,
    step: float = 0.05,
) -> tuple[float | None, float]:
    """Find the smallest multiplier that pushes one feature above the PSI threshold."""
    if feature not in current.columns or feature not in reference.columns:
        return None, float("nan")

    best_psi = 0.0
    multiplier = minimum
    while multiplier <= maximum + 1e-9:
        candidate = current.copy()
        candidate[feature] = apply_multiplier(candidate[feature], multiplier)
        score = psi_numeric(reference[feature], candidate[feature])
        if np.isfinite(score):
            best_psi = max(best_psi, float(score))
            if score >= threshold:
                return float(multiplier), float(score)
        multiplier += step

    return None, float(best_psi)


def positive_scores(model, X: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            classes = getattr(model, "classes_", None)
            if classes is not None:
                classes = list(classes)
                for positive_name in [1, "Yes", "yes", True]:
                    if positive_name in classes:
                        return probabilities[:, classes.index(positive_name)]
            return probabilities[:, 1]
        return probabilities.ravel()
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return scores if np.ndim(scores) == 1 else scores[:, 1]
    return None


def evaluate_model(model, X: pd.DataFrame, y: pd.Series) -> dict:
    predictions = model.predict(X)
    scores = positive_scores(model, X)
    metrics = {
        "Accuracy": float(accuracy_score(y, predictions)),
        "Precision": float(precision_score(y, predictions, zero_division=0)),
        "Recall": float(recall_score(y, predictions, zero_division=0)),
        "F1-score": float(f1_score(y, predictions, zero_division=0)),
    }
    metrics["ROC-AUC"] = float(roc_auc_score(y, scores)) if scores is not None else float("nan")
    return metrics

# Rather than assuming all drifted features impact the business output equally, the system evaluates each feature's contribution to model performance using permutation importance scored against ROC-AUC (Mean_ROC_AUC_Permutation_Importance). 
def compute_feature_contribution(baseline_model, X_reference: pd.DataFrame, y_reference: pd.Series) -> pd.DataFrame:
    print("\nComputing baseline feature contribution...")
    start = time.perf_counter()
    result = permutation_importance(
        baseline_model,
        X_reference,
        y_reference,
        scoring="roc_auc",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    elapsed = time.perf_counter() - start
    contribution = pd.DataFrame(
        {
            "Feature": X_reference.columns.tolist(),
            "Mean_ROC_AUC_Permutation_Importance": result.importances_mean,
            "Std_ROC_AUC_Permutation_Importance": result.importances_std,
        }
    )
    contribution["Meaningful_Contribution"] = (
        contribution["Mean_ROC_AUC_Permutation_Importance"] >= CONTRIBUTION_THRESHOLD
    )
    contribution = contribution.sort_values(
        "Mean_ROC_AUC_Permutation_Importance", ascending=False
    ).reset_index(drop=True)
    print(f"Feature contribution computed in {elapsed:.2f} seconds.")
    return contribution


def full_retraining_trigger(max_psi: float) -> bool:
    return not np.isnan(max_psi) and max_psi >= HIGH_DRIFT_PSI_THRESHOLD

#The call to high_drift.merge(important, on="Feature", how="inner") is the central decision logic. An inner join ensures that retraining is candidate-approved only if a feature exhibits high statistical drift AND holds meaningful predictive weight. 
def selective_retraining_trigger(drift_table: pd.DataFrame, contribution_table: pd.DataFrame):
    if drift_table.empty:
        return False, []
    high_drift = drift_table[drift_table["PSI"] >= HIGH_DRIFT_PSI_THRESHOLD]
    if high_drift.empty:
        return False, []
    important = contribution_table[
        contribution_table["Mean_ROC_AUC_Permutation_Importance"] >= CONTRIBUTION_THRESHOLD
    ][["Feature"]]
    selected = high_drift.merge(important, on="Feature", how="inner")
    selected_features = selected["Feature"].astype(str).tolist()
    return bool(selected_features), selected_features


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("SELF-HEALING ML — SELECTIVE RETRAINING & COST ANALYSIS")
    print("=" * 78)

    df = load_dataset()
    if not BASELINE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Baseline model not found: {BASELINE_MODEL_PATH}")
    baseline_model = joblib.load(BASELINE_MODEL_PATH)

    y = binary_target(df["Churn"])
    split_index = int(len(df) * 0.70)
    reference_df = df.iloc[:split_index].copy()
    current_df = df.iloc[split_index:].copy()
    X_reference = build_features(reference_df)
    y_reference = y.iloc[:split_index].copy()

    contribution = compute_feature_contribution(baseline_model, X_reference, y_reference)
    contribution.to_csv(FEATURE_REPORT_PATH, index=False)

    eligible_low = contribution[
        (~contribution["Feature"].isin(["tenure", "MonthlyCharges"]))
        & (contribution["Mean_ROC_AUC_Permutation_Importance"] < CONTRIBUTION_THRESHOLD)
        & contribution["Feature"].isin(
            [
                c for c in X_reference.select_dtypes(include="number").columns
                if X_reference[c].nunique() >= 10
            ]
        )
    ]
    low_feature = str(eligible_low.iloc[0]["Feature"]) if not eligible_low.empty else "TotalCharges"

    low_feature_multiplier, low_feature_max_psi = find_high_drift_multiplier(
        build_features(reference_df),
        build_features(current_df),
        low_feature,
        HIGH_DRIFT_PSI_THRESHOLD,
    )

    if low_feature_multiplier is None:
        print(
            "\nWARNING: The selected low-contribution feature "
            f"'{low_feature}' did not reach PSI >= {HIGH_DRIFT_PSI_THRESHOLD:.2f} "
            f"within the search range. Maximum achieved PSI: {low_feature_max_psi:.4f}"
        )
    else:
        print(
            "\nHigh-drift low-contribution test: "
            f"{low_feature} x {low_feature_multiplier:.2f} "
            f"-> PSI {low_feature_max_psi:.4f}"
        )

    scenarios = [
        {"Scenario": "No drift", "Description": "No simulated feature shift.", "multipliers": {}},
        {"Scenario": "Mild tenure drift", "Description": "Small tenure shift.", "multipliers": {"tenure": 1.10}},
        {"Scenario": "Mild MonthlyCharges drift", "Description": "Small MonthlyCharges shift.", "multipliers": {"MonthlyCharges": 1.05}},
        {"Scenario": "Moderate tenure drift", "Description": "Moderate tenure shift.", "multipliers": {"tenure": 1.25}},
        {"Scenario": "Strong tenure drift", "Description": "Strong tenure shift.", "multipliers": {"tenure": 1.50}},
        {"Scenario": "Strong MonthlyCharges drift", "Description": "Strong MonthlyCharges shift.", "multipliers": {"MonthlyCharges": 1.20}},
        {"Scenario": "Strong important-feature drift", "Description": "Strong simultaneous drift in two high-contribution features.", "multipliers": {"tenure": 1.50, "MonthlyCharges": 1.20}},
        {
            "Scenario": "High drift / low contribution",
            "Description": "Automatically selected high-drift scenario on a measured low/unclear-contribution feature.",
            "multipliers": (
                {low_feature: low_feature_multiplier}
                if low_feature_multiplier is not None
                else {}
            ),
        },
    ]

    rows = []
    for scenario in scenarios:
        name = scenario["Scenario"]
        print("\n" + "-" * 78)
        print(name)
        print("-" * 78)

        drifted_df = simulate_scenario(current_df, scenario)
        drift_reference = build_features(reference_df)
        drift_current = build_features(drifted_df)
        drift_table = compute_feature_psi(drift_reference, drift_current)
        if drift_table.empty:
            continue

        max_psi = float(drift_table["PSI"].max())
        top_row = drift_table.iloc[0]
        top_feature = str(top_row["Feature"])
        top_feature_psi = float(top_row["PSI"])
        high_drift_features = drift_table.loc[
            drift_table["PSI"] >= HIGH_DRIFT_PSI_THRESHOLD, "Feature"
        ].astype(str).tolist()
# The code explicitly executes two separate decision policies side-by-side: the conventional Full PSI Trigger and the proposed Feature-Aware Selective Trigger. 
        full_trigger = full_retraining_trigger(max_psi)
        selective_trigger, selected_features = selective_retraining_trigger(
            drift_table, contribution
        )

        drifted_with_target = drifted_df.copy()
        drifted_with_target["_target"] = binary_target(drifted_with_target["Churn"])
        adaptation_df, evaluation_df = train_test_split(
            drifted_with_target,
            test_size=0.50,
            random_state=RANDOM_STATE,
            stratify=drifted_with_target["_target"],
        )
        X_adaptation = build_features(adaptation_df)
        y_adaptation = adaptation_df["_target"].astype(int)
        X_evaluation = build_features(evaluation_df)
        y_evaluation = evaluation_df["_target"].astype(int)

        baseline_metrics = evaluate_model(baseline_model, X_evaluation, y_evaluation)

        full_metrics = baseline_metrics.copy()
        full_training_time = 0.0
        if full_trigger:
            candidate = clone(baseline_model)
            start = time.perf_counter()
            candidate.fit(X_adaptation, y_adaptation)
            full_training_time = time.perf_counter() - start
            full_metrics = evaluate_model(candidate, X_evaluation, y_evaluation)

        selective_metrics = baseline_metrics.copy()
        selective_training_time = 0.0
        if selective_trigger:
            selective_candidate = clone(baseline_model)
            start = time.perf_counter()
            selective_candidate.fit(X_adaptation, y_adaptation)
            selective_training_time = time.perf_counter() - start
            selective_metrics = evaluate_model(
                selective_candidate, X_evaluation, y_evaluation
            )

        f1_gain_full = full_metrics["F1-score"] - baseline_metrics["F1-score"]
        auc_gain_full = full_metrics["ROC-AUC"] - baseline_metrics["ROC-AUC"]
        recall_gain_full = full_metrics["Recall"] - baseline_metrics["Recall"]
        unnecessary_full_trigger = full_trigger and not selective_trigger
        missed_opportunity = unnecessary_full_trigger and (
            f1_gain_full > 0.01 or auc_gain_full > 0.01 or recall_gain_full > 0.02
        )

        rows.append(
            {
                "Scenario": name,
                "Description": scenario["Description"],
                "Top_Drift_Feature": top_feature,
                "Top_Feature_PSI": top_feature_psi,
                "Maximum_PSI": max_psi,
                "High_Drift_Features": ", ".join(high_drift_features),
                "Feature_Aware_Drift_Features": ", ".join(selected_features),
                "Full_Retraining_Triggered": full_trigger,
                "Selective_Retraining_Triggered": selective_trigger,
                "Unnecessary_Full_Retraining": unnecessary_full_trigger,
                "Possible_Missed_Retraining_Opportunity": missed_opportunity,
                "Baseline_Accuracy": baseline_metrics["Accuracy"],
                "Baseline_Precision": baseline_metrics["Precision"],
                "Baseline_Recall": baseline_metrics["Recall"],
                "Baseline_F1": baseline_metrics["F1-score"],
                "Baseline_ROC_AUC": baseline_metrics["ROC-AUC"],
                "Full_Retrain_Accuracy": full_metrics["Accuracy"],
                "Full_Retrain_Precision": full_metrics["Precision"],
                "Full_Retrain_Recall": full_metrics["Recall"],
                "Full_Retrain_F1": full_metrics["F1-score"],
                "Full_Retrain_ROC_AUC": full_metrics["ROC-AUC"],
                "Selective_Final_Accuracy": selective_metrics["Accuracy"],
                "Selective_Final_Precision": selective_metrics["Precision"],
                "Selective_Final_Recall": selective_metrics["Recall"],
                "Selective_Final_F1": selective_metrics["F1-score"],
                "Selective_Final_ROC_AUC": selective_metrics["ROC-AUC"],
                "Full_Retrain_Time_Sec": full_training_time,
                "Selective_Retrain_Time_Sec": selective_training_time,
                "F1_Gain_From_Full_Retrain": f1_gain_full,
                "ROC_AUC_Gain_From_Full_Retrain": auc_gain_full,
                "Recall_Gain_From_Full_Retrain": recall_gain_full,
            }
        )

        print(f"Maximum PSI: {max_psi:.4f}")
        print("Full retraining triggered:", full_trigger)
        print("Selective retraining triggered:", selective_trigger)
        print("Feature-aware drift features:", selected_features)

    results_df = pd.DataFrame(rows)

    full_count = int(results_df["Full_Retraining_Triggered"].sum())
    selective_count = int(results_df["Selective_Retraining_Triggered"].sum())
    unnecessary_avoided = int(results_df["Unnecessary_Full_Retraining"].sum())
    missed = int(results_df["Possible_Missed_Retraining_Opportunity"].sum())
    full_time = float(results_df["Full_Retrain_Time_Sec"].sum())
    selective_time = float(results_df["Selective_Retrain_Time_Sec"].sum())

    avoided_training_time = float(
        results_df.loc[
            results_df["Unnecessary_Full_Retraining"],
            "Full_Retrain_Time_Sec",
        ].sum()
    )

    frequency_reduction = (1 - selective_count / full_count) * 100 if full_count else 0.0
    estimated_time_saved = (
        avoided_training_time / full_time * 100
        if full_time
        else 0.0
    )

    summary_df = pd.DataFrame(
        [
            {"Metric": "Scenario count", "Value": len(results_df)},
            {"Metric": "Full-policy retrain count", "Value": full_count},
            {"Metric": "Feature-aware retrain count", "Value": selective_count},
            {"Metric": "Retraining frequency reduction (%)", "Value": frequency_reduction},
            {"Metric": "Estimated training time saved (%)", "Value": estimated_time_saved},
            {"Metric": "Avoided full-retrain time (sec)", "Value": avoided_training_time},
            {"Metric": "Unnecessary full retrains avoided", "Value": unnecessary_avoided},
            {"Metric": "Possible missed retraining opportunities", "Value": missed},
            {"Metric": "Full retraining training time (sec)", "Value": full_time},
            {"Metric": "Selective retraining training time (sec)", "Value": selective_time},
            {"Metric": "PSI threshold", "Value": HIGH_DRIFT_PSI_THRESHOLD},
            {"Metric": "Contribution threshold", "Value": CONTRIBUTION_THRESHOLD},
            {"Metric": "Low-contribution feature used", "Value": low_feature},
        ]
    )

    results_df.to_csv(SCENARIO_REPORT_PATH, index=False)
    summary_df.to_csv(SUMMARY_REPORT_PATH, index=False)

    config = {
        "random_state": RANDOM_STATE,
        "high_drift_psi_threshold": HIGH_DRIFT_PSI_THRESHOLD,
        "contribution_threshold": CONTRIBUTION_THRESHOLD,
        "dataset": str(DATA_PATH.relative_to(PROJECT_ROOT)),
        "baseline_model": str(BASELINE_MODEL_PATH.relative_to(PROJECT_ROOT)),
        "policy_full": "Retrain when maximum PSI >= threshold",
        "policy_selective": "Retrain only when high drift overlaps with meaningful feature contribution",
        "low_contribution_feature": low_feature,
        "high_drift_low_contribution_multiplier": low_feature_multiplier,
        "high_drift_low_contribution_max_or_achieved_psi": low_feature_max_psi,
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=4), encoding="utf-8")

    with TEXT_REPORT_PATH.open("w", encoding="utf-8") as file:
        file.write("SELF-HEALING ML — SELECTIVE RETRAINING & COST ANALYSIS\n")
        file.write("=" * 70 + "\n\n")
        file.write(
            "Research question:\n"
            "Can feature-level drift diagnosis reduce unnecessary retraining "
            "while maintaining or improving predictive performance?\n\n"
        )
        file.write("Summary:\n")
        for row in summary_df.to_dict("records"):
            file.write(f"- {row['Metric']}: {row['Value']}\n")
        file.write("\nScenario-level results:\n")
        file.write(results_df.to_string(index=False))
        file.write(
            "\n\nInterpretation:\n"
            "This is a controlled offline experiment. Training time is used "
            "as a practical cost proxy. PSI and permutation-importance thresholds "
            "are experiment-specific heuristics and should not be treated as universal.\n"
        )

    print("\n" + "=" * 78)
    print("EXPERIMENT COMPLETED")
    print("=" * 78)
    print("Scenario report:", SCENARIO_REPORT_PATH.relative_to(PROJECT_ROOT))
    print("Summary report:", SUMMARY_REPORT_PATH.relative_to(PROJECT_ROOT))
    print("Feature contribution:", FEATURE_REPORT_PATH.relative_to(PROJECT_ROOT))
    print(f"Retraining frequency reduction: {frequency_reduction:.2f}%")
    print(f"Estimated training time saved from avoided retrains: {estimated_time_saved:.2f}%")
    print(f"Avoided full-retrain time: {avoided_training_time:.4f} sec")
    print("Unnecessary full retrains avoided:", unnecessary_avoided)
    print("Possible missed opportunities:", missed)


if __name__ == "__main__":
    main()
