
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from self_healing_pipeline import (
    MODELS_DIR,
    REPORTS_DIR,
    load_data,
    train_test_split,
    RANDOM_STATE,
    apply_synthetic_drift,
)


# ============================================================
# 1. HELPERS
# ============================================================

def get_transformed_feature_names(model):
    """Get feature names after preprocessing."""

    preprocessor = model.named_steps["preprocessor"]

    try:
        return np.asarray(
            preprocessor.get_feature_names_out(),
            dtype=str,
        )
    except (AttributeError, ValueError):
        feature_count = model.named_steps["classifier"].coef_.shape[1]

        return np.asarray(
            [f"transformed_feature_{i}" for i in range(feature_count)],
            dtype=str,
        )


def extract_coefficients(model):
    """
    Extract Logistic Regression coefficients.

    Positive coefficients increase the model's log-odds
    for the positive class; negative coefficients decrease it.
    """

    classifier = model.named_steps["classifier"]
    feature_names = get_transformed_feature_names(model)
    coefficients = classifier.coef_[0]

    if len(feature_names) != len(coefficients):
        raise ValueError(
            "The number of transformed feature names does not match "
            "the number of model coefficients."
        )

    result = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "absolute_coefficient": np.abs(coefficients),
    })

    result["direction"] = np.where(
        result["coefficient"] > 0,
        "Positive association with predicted churn",
        np.where(
            result["coefficient"] < 0,
            "Negative association with predicted churn",
            "No linear association",
        ),
    )

    return result.sort_values(
        "absolute_coefficient",
        ascending=False,
    ).reset_index(drop=True)


def explain_single_prediction(model, row):
    """
    Provide a coefficient-based contribution summary
    for one observation.

    Contributions are calculated in transformed feature space.
    They are additive contributions to the model's log-odds,
    not direct probability changes.
    """

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    transformed = preprocessor.transform(row)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    values = np.asarray(transformed).ravel()
    coefficients = classifier.coef_[0]
    intercept = float(classifier.intercept_[0])
    feature_names = get_transformed_feature_names(model)

    if not (
        len(values) == len(coefficients) == len(feature_names)
    ):
        raise ValueError(
            "Mismatch between transformed values, coefficients, "
            "and feature names."
        )

    contributions = values * coefficients

    explanation = pd.DataFrame({
        "feature": feature_names,
        "transformed_value": values,
        "coefficient": coefficients,
        "contribution_to_log_odds": contributions,
        "absolute_contribution": np.abs(contributions),
    })

    explanation = explanation.sort_values(
        "absolute_contribution",
        ascending=False,
    ).reset_index(drop=True)

    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0, 1])

    return {
        "prediction": prediction,
        "predicted_churn_probability": probability,
        "intercept": intercept,
        "explanation": explanation,
    }


# ============================================================
# 2. MAIN XAI EXPERIMENT
# ============================================================

def main():
    print("=" * 70)
    print("EXPLAINABLE AI: BASELINE VS CANDIDATE")
    print("=" * 70)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    baseline_path = MODELS_DIR / "integrated_baseline_model.joblib"
    candidate_path = MODELS_DIR / "integrated_candidate_model.joblib"

    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Baseline model not found: {baseline_path}\n"
            "Run src/self_healing_pipeline.py first."
        )

    if not candidate_path.exists():
        raise FileNotFoundError(
            f"Candidate model not found: {candidate_path}\n"
            "Run src/self_healing_pipeline.py first."
        )

    baseline_model = joblib.load(baseline_path)
    candidate_model = joblib.load(candidate_path)

    # --------------------------------------------------------
    # A. Global coefficient explanations
    # --------------------------------------------------------

    baseline_global = extract_coefficients(baseline_model)
    candidate_global = extract_coefficients(candidate_model)

    baseline_global["model"] = "baseline"
    candidate_global["model"] = "candidate"

    global_comparison = pd.concat(
        [baseline_global, candidate_global],
        ignore_index=True,
    )

    global_path = REPORTS_DIR / "xai_global_coefficients.csv"
    global_comparison.to_csv(global_path, index=False)

    # --------------------------------------------------------
    # B. Compare feature coefficients by name
    # --------------------------------------------------------

    coefficient_comparison = baseline_global[
        ["feature", "coefficient"]
    ].merge(
        candidate_global[["feature", "coefficient"]],
        on="feature",
        how="outer",
        suffixes=("_baseline", "_candidate"),
    ).fillna(0)

    coefficient_comparison["coefficient_change"] = (
        coefficient_comparison["coefficient_candidate"]
        - coefficient_comparison["coefficient_baseline"]
    )

    coefficient_comparison["absolute_change"] = (
        coefficient_comparison["coefficient_change"].abs()
    )

    coefficient_comparison = coefficient_comparison.sort_values(
        "absolute_change",
        ascending=False,
    ).reset_index(drop=True)

    comparison_path = REPORTS_DIR / "xai_coefficient_comparison.csv"
    coefficient_comparison.to_csv(comparison_path, index=False)

    # --------------------------------------------------------
    # C. Explain one held-out shifted observation
    # --------------------------------------------------------

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_adapt, X_eval, _, _ = train_test_split(
        X_test,
        y_test,
        test_size=0.50,
        random_state=21,
        stratify=y_test,
    )

    X_eval_shifted = apply_synthetic_drift(X_eval)

    if len(X_eval_shifted) == 0:
        raise ValueError("No evaluation rows are available to explain.")

    # Explain the first row in the shifted evaluation partition.
    sample = X_eval_shifted.iloc[[0]]

    baseline_explanation = explain_single_prediction(
        baseline_model,
        sample,
    )

    candidate_explanation = explain_single_prediction(
        candidate_model,
        sample,
    )

    sample_rows = []

    for model_name, result in [
        ("baseline", baseline_explanation),
        ("candidate", candidate_explanation),
    ]:
        top = result["explanation"].head(10).copy()
        top["model"] = model_name
        top["predicted_churn"] = result["prediction"]
        top["predicted_churn_probability"] = (
            result["predicted_churn_probability"]
        )
        sample_rows.append(top)

    sample_comparison = pd.concat(
        sample_rows,
        ignore_index=True,
    )

    sample_path = REPORTS_DIR / "xai_single_prediction.csv"
    sample_comparison.to_csv(sample_path, index=False)

    # --------------------------------------------------------
    # D. Human-readable summary
    # --------------------------------------------------------

    summary = {
        "explanation_method": (
            "Logistic Regression coefficients and transformed-space "
            "linear contributions"
        ),
        "baseline_top_features": (
            baseline_global.head(10)["feature"].tolist()
        ),
        "candidate_top_features": (
            candidate_global.head(10)["feature"].tolist()
        ),
        "sample_baseline_prediction": (
            baseline_explanation["prediction"]
        ),
        "sample_baseline_churn_probability": (
            baseline_explanation["predicted_churn_probability"]
        ),
        "sample_candidate_prediction": (
            candidate_explanation["prediction"]
        ),
        "sample_candidate_churn_probability": (
            candidate_explanation["predicted_churn_probability"]
        ),
        "limitations": [
            (
                "Coefficients describe the fitted logistic model and "
                "are not proof of causation."
            ),
            (
                "One-hot encoded categories appear as transformed "
                "features rather than being automatically grouped."
            ),
            (
                "Single-record contributions are on the log-odds scale, "
                "not direct probability changes."
            ),
            (
                "The explained observation is from a synthetic drift "
                "experiment, not live production data."
            ),
        ],
    }

    summary_path = REPORTS_DIR / "xai_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    # --------------------------------------------------------
    # E. Display key findings
    # --------------------------------------------------------

    print("\nTop baseline coefficients:")
    print(
        baseline_global[
            ["feature", "coefficient", "direction"]
        ].head(10).to_string(index=False)
    )

    print("\nTop candidate coefficients:")
    print(
        candidate_global[
            ["feature", "coefficient", "direction"]
        ].head(10).to_string(index=False)
    )

    print("\nSingle-record comparison:")
    print(f"Baseline prediction: {baseline_explanation['prediction']}")
    print(
        "Baseline churn probability: "
        f"{baseline_explanation['predicted_churn_probability']:.4f}"
    )
    print(f"Candidate prediction: {candidate_explanation['prediction']}")
    print(
        "Candidate churn probability: "
        f"{candidate_explanation['predicted_churn_probability']:.4f}"
    )

    print("\nSaved reports:")
    print(global_path)
    print(comparison_path)
    print(sample_path)
    print(summary_path)


if __name__ == "__main__":
    main()