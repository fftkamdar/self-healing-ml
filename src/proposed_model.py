"""
SH-DAE
------
Self-Healing Drift-Aware Ensemble

This experiment extends the five-model benchmark by creating an
adaptive ensemble whose model weights depend on:

1. Model performance on adaptation data.
2. Feature-level permutation importance.
3. Exposure of each model to drifted features.

The final evaluation is performed on a separate held-out dataset.

Models:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Machine
- SH-DAE (proposed model)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Telco-Customer-Churn.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "proposed_model"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_STATE = 42

# We use:
# 60% training
# 20% adaptation
# 20% final evaluation

TRAIN_SIZE = 0.60
ADAPTATION_SIZE = 0.20
EVALUATION_SIZE = 0.20

N_PERMUTATION_REPEATS = 7

# Controls how strongly drift exposure affects model weight.
DRIFT_PENALTY = 2.0

EPSILON = 1e-6


# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load and prepare the Telco Customer Churn dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    # Convert TotalCharges to numeric.
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce",
    )

    # Remove identifier.
    if "customerID" in df.columns:
        df = df.drop(
            columns=["customerID"]
        )

    # Encode target.
    if "Churn" not in df.columns:
        raise ValueError(
            "Target column 'Churn' was not found."
        )

    y = (
        df["Churn"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "yes": 1,
                "no": 0,
            }
        )
    )

    if y.isna().any():
        raise ValueError(
            "Unexpected values found in Churn."
        )

    X = df.drop(
        columns=["Churn"]
    )

    return X, y.astype(int)


# =============================================================================
# PREPROCESSOR
# =============================================================================

def create_preprocessor(
    X: pd.DataFrame,
) -> ColumnTransformer:
    """Create common preprocessing for all five models."""

    numeric_features = (
        X.select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    categorical_features = (
        X.select_dtypes(
            include=["object", "category", "string"]
        )
        .columns
        .tolist()
    )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )


# =============================================================================
# FIVE BASE MODELS
# =============================================================================

def build_models() -> Dict[str, object]:
    """Return the five benchmark models."""

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
        "Support Vector Machine": SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


# =============================================================================
# DRIFT SIMULATION
# =============================================================================

def apply_controlled_drift(
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply the same controlled drift used elsewhere in this project.

    Important:
    This represents simulated drift for an academic experiment.
    """

    shifted = X.copy()

    if "tenure" in shifted.columns:
        shifted["tenure"] = (
            shifted["tenure"] * 1.5
        ).clip(
            upper=72
        )

    if "MonthlyCharges" in shifted.columns:
        shifted["MonthlyCharges"] = (
            shifted["MonthlyCharges"] * 1.2
        ).clip(
            upper=120
        )

    return shifted


# =============================================================================
# PSI
# =============================================================================

def calculate_numeric_psi(
    reference: pd.Series,
    current: pd.Series,
) -> float:
    """
    Calculate PSI using bins derived only from reference data.
    """

    reference = pd.to_numeric(
        reference,
        errors="coerce",
    )

    current = pd.to_numeric(
        current,
        errors="coerce",
    )

    reference = reference.dropna()
    current = current.dropna()

    if reference.empty or current.empty:
        return 0.0

    quantiles = np.linspace(
        0,
        1,
        11,
    )

    boundaries = np.unique(
        reference.quantile(
            quantiles
        ).to_numpy()
    )

    if len(boundaries) < 2:
        return 0.0

    # Expand boundaries slightly so the extreme values are included.
    boundaries[0] = -np.inf
    boundaries[-1] = np.inf

    reference_bins = pd.cut(
        reference,
        bins=boundaries,
        include_lowest=True,
    )

    current_bins = pd.cut(
        current,
        bins=boundaries,
        include_lowest=True,
    )

    reference_proportions = (
        reference_bins.value_counts(
            normalize=True,
            sort=False,
        )
        .to_numpy()
    )

    current_proportions = (
        current_bins.value_counts(
            normalize=True,
            sort=False,
        )
        .to_numpy()
    )

    reference_proportions = np.clip(
        reference_proportions,
        EPSILON,
        None,
    )

    current_proportions = np.clip(
        current_proportions,
        EPSILON,
        None,
    )

    psi = np.sum(
        (
            current_proportions
            - reference_proportions
        )
        * np.log(
            current_proportions
            / reference_proportions
        )
    )

    return float(psi)


def calculate_drift_table(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate PSI for every numeric feature."""

    rows = []

    for feature in reference.columns:

        # We calculate numeric PSI where possible.
        if pd.api.types.is_numeric_dtype(
            reference[feature]
        ):
            psi = calculate_numeric_psi(
                reference[feature],
                current[feature],
            )
        else:
            # For categorical columns, compare category proportions.
            reference_counts = (
                reference[feature]
                .astype(str)
                .value_counts(
                    normalize=True
                )
            )

            current_counts = (
                current[feature]
                .astype(str)
                .value_counts(
                    normalize=True
                )
            )

            categories = sorted(
                set(reference_counts.index)
                | set(current_counts.index)
            )

            ref_values = np.array(
                [
                    reference_counts.get(
                        category,
                        0.0,
                    )
                    for category in categories
                ]
            )

            cur_values = np.array(
                [
                    current_counts.get(
                        category,
                        0.0,
                    )
                    for category in categories
                ]
            )

            ref_values = np.clip(
                ref_values,
                EPSILON,
                None,
            )

            cur_values = np.clip(
                cur_values,
                EPSILON,
                None,
            )

            ref_values /= ref_values.sum()
            cur_values /= cur_values.sum()

            psi = float(
                np.sum(
                    (
                        cur_values
                        - ref_values
                    )
                    * np.log(
                        cur_values
                        / ref_values
                    )
                )
            )

        rows.append(
            {
                "Feature": feature,
                "PSI": psi,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# TRAIN A MODEL
# =============================================================================

def train_pipeline(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
):
    """Create, fit and return a complete preprocessing+model pipeline."""

    preprocessor = create_preprocessor(
        X_train
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    return pipeline


# =============================================================================
# PERMUTATION IMPORTANCE
# =============================================================================

def calculate_feature_importance(
    pipeline,
    X_adaptation: pd.DataFrame,
    y_adaptation: pd.Series,
) -> pd.DataFrame:
    """Calculate original-feature permutation importance."""

    result = permutation_importance(
        pipeline,
        X_adaptation,
        y_adaptation,
        scoring="roc_auc",
        n_repeats=N_PERMUTATION_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return pd.DataFrame(
        {
            "Feature": X_adaptation.columns,
            "Importance": result.importances_mean,
            "Importance_Std": result.importances_std,
        }
    )


# =============================================================================
# MODEL PERFORMANCE
# =============================================================================

def evaluate_predictions(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> dict:
    """Calculate all requested classification metrics."""

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return {
        "Accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "F1-score": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "ROC-AUC": roc_auc_score(
            y_true,
            probabilities,
        ),
    }


# =============================================================================
# BUILD DRIFT-AWARE WEIGHTS
# =============================================================================

def calculate_model_weights(
    model_information: dict,
    drift_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate adaptive ensemble weights.

    Weight logic:

    1. Start with adaptation ROC-AUC.
    2. Calculate each model's exposure to drifted features.
    3. Penalize models with greater drift exposure.
    4. Normalize the resulting stability scores.

    This is a project-specific adaptive weighting strategy.
    """

    drift_lookup = dict(
        zip(
            drift_table["Feature"],
            drift_table["PSI"],
        )
    )

    rows = []

    for model_name, info in model_information.items():

        importance_df = info[
            "importance"
        ].copy()

        importance_df["PSI"] = (
            importance_df["Feature"]
            .map(drift_lookup)
            .fillna(0.0)
        )

        # We use absolute importance because a feature can affect
        # the model in either direction.
        importance_df["Absolute_Importance"] = (
            importance_df["Importance"]
            .abs()
        )

        total_importance = (
            importance_df[
                "Absolute_Importance"
            ].sum()
        )

        if total_importance <= EPSILON:
            drift_exposure = 0.0
        else:
            normalized_importance = (
                importance_df[
                    "Absolute_Importance"
                ]
                / total_importance
            )

            drift_exposure = float(
                (
                    normalized_importance
                    * importance_df["PSI"]
                ).sum()
            )

        adaptation_auc = info[
            "adaptation_auc"
        ]

        # Keep scores positive.
        performance_score = max(
            float(adaptation_auc),
            EPSILON,
        )

        stability_score = (
            performance_score
            / (
                1.0
                + DRIFT_PENALTY
                * drift_exposure
            )
        )

        rows.append(
            {
                "Model": model_name,
                "Adaptation_ROC_AUC": adaptation_auc,
                "Drift_Exposure": drift_exposure,
                "Stability_Score": stability_score,
            }
        )

    weights_df = pd.DataFrame(
        rows
    )

    total_score = (
        weights_df[
            "Stability_Score"
        ].sum()
    )

    if total_score <= EPSILON:
        weights_df[
            "Weight"
        ] = 1.0 / len(weights_df)
    else:
        weights_df[
            "Weight"
        ] = (
            weights_df[
                "Stability_Score"
            ]
            / total_score
        )

    return weights_df.sort_values(
        "Weight",
        ascending=False,
    ).reset_index(drop=True)


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main() -> None:

    print("=" * 75)
    print("SELF-HEALING ML — SH-DAE PROPOSED MODEL")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    X, y = load_dataset()

    print(
        f"\nDataset: {X.shape[0]} rows, "
        f"{X.shape[1]} features"
    )

    # -------------------------------------------------------------------------
    # First split: training vs remaining
    # -------------------------------------------------------------------------

    X_train, X_remaining, y_train, y_remaining = (
        train_test_split(
            X,
            y,
            test_size=(
                ADAPTATION_SIZE
                + EVALUATION_SIZE
            ),
            stratify=y,
            random_state=RANDOM_STATE,
        )
    )

    # -------------------------------------------------------------------------
    # Second split: adaptation vs final evaluation
    # -------------------------------------------------------------------------

    evaluation_fraction_of_remaining = (
        EVALUATION_SIZE
        / (
            ADAPTATION_SIZE
            + EVALUATION_SIZE
        )
    )

    X_adaptation, X_eval, y_adaptation, y_eval = (
        train_test_split(
            X_remaining,
            y_remaining,
            test_size=evaluation_fraction_of_remaining,
            stratify=y_remaining,
            random_state=RANDOM_STATE,
        )
    )

    print(
        f"\nTraining records: {len(X_train)}"
    )
    print(
        f"Adaptation records: {len(X_adaptation)}"
    )
    print(
        f"Evaluation records: {len(X_eval)}"
    )

    # -------------------------------------------------------------------------
    # Controlled drift for adaptation and evaluation data
    # -------------------------------------------------------------------------

    X_adaptation_drifted = apply_controlled_drift(
        X_adaptation
    )

    X_eval_drifted = apply_controlled_drift(
        X_eval
    )

    print(
        "\nControlled drift applied to:"
    )
    print("  - tenure")
    print("  - MonthlyCharges")

    # -------------------------------------------------------------------------
    # Calculate drift using TRAINING as reference
    # -------------------------------------------------------------------------

    drift_table = calculate_drift_table(
        X_train,
        X_adaptation_drifted,
    )

    drift_path = (
        REPORT_DIR
        / "proposed_model_drift.csv"
    )

    drift_table.to_csv(
        drift_path,
        index=False,
    )

    print("\nTop drifted features:")

    print(
        drift_table
        .sort_values(
            "PSI",
            ascending=False,
        )
        .head(10)
        .to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Train the five existing models
    # -------------------------------------------------------------------------

    models = build_models()

    model_pipelines = {}
    model_information = {}
    final_probabilities = {}

    print(
        "\n" + "=" * 75
    )
    print("TRAINING FIVE BASE MODELS")
    print("=" * 75)

    for model_name, model in models.items():

        print(
            f"\nTraining {model_name}..."
        )

        pipeline = train_pipeline(
            model,
            X_train,
            y_train,
        )

        model_pipelines[
            model_name
        ] = pipeline

        # Adaptation performance.
        adaptation_probabilities = (
            pipeline.predict_proba(
                X_adaptation_drifted
            )[:, 1]
        )

        adaptation_auc = (
            roc_auc_score(
                y_adaptation,
                adaptation_probabilities,
            )
        )

        # Feature importance.
        importance_df = (
            calculate_feature_importance(
                pipeline,
                X_adaptation_drifted,
                y_adaptation,
            )
        )

        model_information[
            model_name
        ] = {
            "adaptation_auc": adaptation_auc,
            "importance": importance_df,
        }

        # Final evaluation probabilities.
        final_probabilities[
            model_name
        ] = pipeline.predict_proba(
            X_eval_drifted
        )[:, 1]

        print(
            f"Adaptation ROC-AUC: "
            f"{adaptation_auc:.6f}"
        )

    # -------------------------------------------------------------------------
    # Calculate adaptive weights
    # -------------------------------------------------------------------------

    weights_df = calculate_model_weights(
        model_information,
        drift_table,
    )

    print(
        "\n" + "=" * 75
    )
    print("ADAPTIVE MODEL WEIGHTS")
    print("=" * 75)

    print(
        weights_df.to_string(
            index=False
        )
    )

    weights_path = (
        REPORT_DIR
        / "sh_dae_model_weights.csv"
    )

    weights_df.to_csv(
        weights_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # SH-DAE prediction
    # -------------------------------------------------------------------------

    weight_lookup = dict(
        zip(
            weights_df["Model"],
            weights_df["Weight"],
        )
    )

    dae_probability = np.zeros(
        len(X_eval_drifted)
    )

    for model_name in models:

        dae_probability += (
            weight_lookup[model_name]
            * final_probabilities[
                model_name
            ]
        )

    final_probabilities[
        "SH-DAE (Proposed)"
    ] = dae_probability

    # -------------------------------------------------------------------------
    # Evaluate all six models
    # -------------------------------------------------------------------------

    results = []

    for model_name, probabilities in (
        final_probabilities.items()
    ):

        metrics = evaluate_predictions(
            y_eval,
            probabilities,
        )

        results.append(
            {
                "Model": model_name,
                **metrics,
            }
        )

    results_df = pd.DataFrame(
        results
    )

    # Add information about model weighting only to SH-DAE.
    results_df[
        "Drift_Aware"
    ] = results_df[
        "Model"
    ].eq(
        "SH-DAE (Proposed)"
    )

    results_path = (
        REPORT_DIR
        / "six_model_final_comparison.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Print final results
    # -------------------------------------------------------------------------

    print(
        "\n" + "=" * 75
    )
    print("FINAL SIX-MODEL COMPARISON")
    print("=" * 75)

    print(
        results_df.to_string(
            index=False,
        )
    )

    # -------------------------------------------------------------------------
    # Calculate deltas relative to best conventional benchmark
    # -------------------------------------------------------------------------

    conventional = results_df[
        results_df["Model"]
        != "SH-DAE (Proposed)"
    ]

    proposed = results_df[
        results_df["Model"]
        == "SH-DAE (Proposed)"
    ].iloc[0]

    comparison_rows = []

    for metric in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
    ]:

        conventional_best = (
            conventional[metric].max()
        )

        proposed_value = (
            proposed[metric]
        )

        comparison_rows.append(
            {
                "Metric": metric,
                "Best_Conventional_Value": (
                    conventional_best
                ),
                "SH_DAE_Value": (
                    proposed_value
                ),
                "SH_DAE_Delta": (
                    proposed_value
                    - conventional_best
                ),
            }
        )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_path = (
        REPORT_DIR
        / "sh_dae_vs_conventional.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    print(
        "\n" + "=" * 75
    )
    print("SH-DAE DELTA AGAINST BEST CONVENTIONAL VALUE")
    print("=" * 75)

    print(
        comparison_df.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Plot six-model ROC-AUC comparison
    # -------------------------------------------------------------------------

    plt.figure(
        figsize=(11, 6)
    )

    plt.bar(
        results_df["Model"],
        results_df["ROC-AUC"],
    )

    plt.ylabel(
        "ROC-AUC"
    )

    plt.xlabel(
        "Model"
    )

    plt.title(
        "Conventional Models vs SH-DAE"
    )

    plt.xticks(
        rotation=30,
        ha="right",
    )

    plt.tight_layout()

    chart_path = (
        REPORT_DIR
        / "six_model_roc_auc.png"
    )

    plt.savefig(
        chart_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # -------------------------------------------------------------------------
    # Save JSON metadata
    # -------------------------------------------------------------------------

    metadata = {
        "model_name": "SH-DAE",
        "full_name": (
            "Self-Healing Drift-Aware Ensemble"
        ),
        "base_models": list(
            models.keys()
        ),
        "random_state": RANDOM_STATE,
        "drift_penalty": DRIFT_PENALTY,
        "permutation_repeats": (
            N_PERMUTATION_REPEATS
        ),
        "drifted_features": [
            "tenure",
            "MonthlyCharges",
        ],
        "evaluation_note": (
            "Final metrics are measured on a "
            "separate held-out drifted evaluation set. "
            "Adaptive weights are determined before "
            "final evaluation using training and "
            "adaptation data."
        ),
    }

    metadata_path = (
        REPORT_DIR
        / "sh_dae_summary.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Finish
    # -------------------------------------------------------------------------

    print(
        "\n" + "=" * 75
    )
    print("SH-DAE EXPERIMENT COMPLETED")
    print("=" * 75)

    print(
        f"\nReports saved to:\n{REPORT_DIR}"
    )


if __name__ == "__main__":
    main()