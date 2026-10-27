"""
SH-DAS V3
---------
Self-Healing Drift-Aware Stacking

Purpose
-------
Build a proposed model that learns how to combine five conventional
classifiers while explicitly receiving information about feature drift.

Base models:
    1. Logistic Regression
    2. Decision Tree
    3. Random Forest
    4. Gradient Boosting
    5. Support Vector Machine

Meta-model:
    Logistic Regression

The meta-model receives:
    - probability produced by each base model
    - drift-aware feature values for the main drifted features
    - interactions between model probabilities and drift intensity

Evaluation protocol:
    Training data:
        fit five base models

    Adaptation data:
        generate base predictions
        calculate drift-aware meta-features
        train meta-model
        select operating threshold

    Final evaluation data:
        NEVER used for meta-model training or threshold selection

This is an academic prototype and not a production deployment system.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
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
    / "proposed_model_v3"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_STATE = 42

ADAPTATION_SIZE = 0.20
EVALUATION_SIZE = 0.20

EPSILON = 1e-6

# These are the two features deliberately drifted in the project's
# existing controlled-drift experiment.
DRIFT_FEATURES = [
    "tenure",
    "MonthlyCharges",
]

# Threshold search is performed only on adaptation data.
THRESHOLDS = np.arange(
    0.20,
    0.81,
    0.01,
)


# =============================================================================
# LOAD DATA
# =============================================================================

def load_dataset() -> tuple[pd.DataFrame, pd.Series]:

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH
    )

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce",
    )

    if "customerID" in df.columns:
        df = df.drop(
            columns=["customerID"]
        )

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
            "Unexpected Churn values were found."
        )

    X = df.drop(
        columns=["Churn"]
    )

    return X, y.astype(int)


# =============================================================================
# PREPROCESSING
# =============================================================================

def create_preprocessor(
    X: pd.DataFrame,
) -> ColumnTransformer:

    numeric_features = (
        X.select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    categorical_features = (
        X.select_dtypes(
            include=[
                "object",
                "category",
                "string",
            ]
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
# BASE MODELS
# =============================================================================

def build_base_models():

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
# CONTROLLED DRIFT
# =============================================================================

def apply_controlled_drift(
    X: pd.DataFrame,
) -> pd.DataFrame:

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
# DRIFT INTENSITY FEATURES
# =============================================================================

def calculate_drift_reference(
    X_train: pd.DataFrame,
) -> dict:

    reference = {}

    for feature in DRIFT_FEATURES:

        if feature not in X_train.columns:
            continue

        values = pd.to_numeric(
            X_train[feature],
            errors="coerce",
        )

        reference[feature] = {
            "mean": float(
                values.mean()
            ),
            "std": float(
                values.std()
            ),
        }

    return reference


def create_drift_meta_features(
    X: pd.DataFrame,
    reference: dict,
) -> pd.DataFrame:
    """
    Create row-level drift indicators.

    For every drifted feature:
        signed deviation
        absolute deviation

    These are calculated relative to the training distribution.
    """

    features = pd.DataFrame(
        index=X.index
    )

    for feature in DRIFT_FEATURES:

        if feature not in X.columns:
            continue

        if feature not in reference:
            continue

        values = pd.to_numeric(
            X[feature],
            errors="coerce",
        )

        mean = reference[
            feature
        ]["mean"]

        std = reference[
            feature
        ]["std"]

        std = max(
            std,
            EPSILON,
        )

        z = (
            values - mean
        ) / std

        z = z.fillna(0.0)

        features[
            f"{feature}_drift_z"
        ] = z

        features[
            f"{feature}_drift_abs"
        ] = z.abs()

    return features


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold=0.50,
):

    predictions = (
        probabilities
        >= threshold
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
# BASE MODEL TRAINING
# =============================================================================

def train_base_model(
    model,
    X_train,
    y_train,
):

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
# META-DATASET
# =============================================================================

def create_meta_dataset(
    base_models: dict,
    pipelines: dict,
    X: pd.DataFrame,
    drift_reference: dict,
) -> pd.DataFrame:
    """
    Construct meta-model inputs.

    Columns include:
        model probability
        drift-aware row features
        probability × drift interactions
    """

    meta = pd.DataFrame(
        index=X.index
    )

    drift_features = (
        create_drift_meta_features(
            X,
            drift_reference,
        )
    )

    # Add row-level drift indicators.
    for column in drift_features.columns:
        meta[column] = drift_features[
            column
        ].values

    # Add each model's predicted probability.
    probabilities = {}

    for model_name in base_models:

        prob = (
            pipelines[
                model_name
            ]
            .predict_proba(
                X
            )[:, 1]
        )

        safe_name = (
            model_name
            .replace(" ", "_")
            .replace("-", "_")
        )

        column_name = (
            f"{safe_name}_prob"
        )

        meta[
            column_name
        ] = prob

        probabilities[
            model_name
        ] = prob

    # Add model probability × drift interactions.
    for model_name in base_models:

        safe_name = (
            model_name
            .replace(" ", "_")
            .replace("-", "_")
        )

        probability_column = (
            f"{safe_name}_prob"
        )

        for drift_column in (
            drift_features.columns
        ):

            interaction_name = (
                f"{safe_name}_x_"
                f"{drift_column}"
            )

            meta[
                interaction_name
            ] = (
                meta[
                    probability_column
                ]
                * meta[
                    drift_column
                ]
            )

    return meta


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print(
        "SELF-HEALING ML — SH-DAS V3"
    )
    print(
        "DRIFT-AWARE STACKING"
    )
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------

    X, y = load_dataset()

    print(
        f"\nDataset: {len(X)} rows"
    )

    print(
        f"Original features: {X.shape[1]}"
    )

    # -------------------------------------------------------------------------
    # Three-way split
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

    evaluation_fraction = (
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
            test_size=evaluation_fraction,
            stratify=y_remaining,
            random_state=RANDOM_STATE,
        )
    )

    print(
        f"\nTraining records:   {len(X_train)}"
    )

    print(
        f"Adaptation records: {len(X_adaptation)}"
    )

    print(
        f"Evaluation records: {len(X_eval)}"
    )

    # -------------------------------------------------------------------------
    # Simulated drift
    # -------------------------------------------------------------------------

    X_adaptation_drifted = (
        apply_controlled_drift(
            X_adaptation
        )
    )

    X_eval_drifted = (
        apply_controlled_drift(
            X_eval
        )
    )

    print(
        "\nControlled drift:"
    )

    print(
        "  tenure × 1.5, capped at 72"
    )

    print(
        "  MonthlyCharges × 1.2, capped at 120"
    )

    # -------------------------------------------------------------------------
    # Drift reference
    # -------------------------------------------------------------------------

    drift_reference = (
        calculate_drift_reference(
            X_train
        )
    )

    # -------------------------------------------------------------------------
    # Train five models
    # -------------------------------------------------------------------------

    base_models = build_base_models()

    pipelines = {}

    print(
        "\n" + "=" * 80
    )

    print(
        "TRAINING FIVE BASE MODELS"
    )

    print(
        "=" * 80
    )

    for model_name, model in (
        base_models.items()
    ):

        print(
            f"\nTraining {model_name}..."
        )

        pipeline = train_base_model(
            model,
            X_train,
            y_train,
        )

        pipelines[
            model_name
        ] = pipeline

        adaptation_probabilities = (
            pipeline.predict_proba(
                X_adaptation_drifted
            )[:, 1]
        )

        adaptation_metrics = (
            calculate_metrics(
                y_adaptation,
                adaptation_probabilities,
                threshold=0.50,
            )
        )

        print(
            f"Adaptation F1: "
            f"{adaptation_metrics['F1-score']:.6f}"
        )

        print(
            f"Adaptation Recall: "
            f"{adaptation_metrics['Recall']:.6f}"
        )

        print(
            f"Adaptation ROC-AUC: "
            f"{adaptation_metrics['ROC-AUC']:.6f}"
        )

    # -------------------------------------------------------------------------
    # Create adaptation meta dataset
    # -------------------------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "BUILDING META-DATASET"
    )

    print(
        "=" * 80
    )

    adaptation_meta = create_meta_dataset(
        base_models,
        pipelines,
        X_adaptation_drifted,
        drift_reference,
    )

    evaluation_meta = create_meta_dataset(
        base_models,
        pipelines,
        X_eval_drifted,
        drift_reference,
    )

    print(
        f"\nMeta-feature count: "
        f"{adaptation_meta.shape[1]}"
    )

    print(
        f"Adaptation meta shape: "
        f"{adaptation_meta.shape}"
    )

    print(
        f"Evaluation meta shape: "
        f"{evaluation_meta.shape}"
    )

    # Save meta datasets for research transparency.
    adaptation_meta.to_csv(
        REPORT_DIR
        / "adaptation_meta_features.csv",
        index=False,
    )

    evaluation_meta.to_csv(
        REPORT_DIR
        / "evaluation_meta_features.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Train meta-model
    # -------------------------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "TRAINING SH-DAS META-MODEL"
    )

    print(
        "=" * 80
    )

    meta_model = Pipeline(
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
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    meta_model.fit(
        adaptation_meta,
        y_adaptation,
    )

    adaptation_meta_probability = (
        meta_model.predict_proba(
            adaptation_meta
        )[:, 1]
    )

    # -------------------------------------------------------------------------
    # Threshold search
    # -------------------------------------------------------------------------

    threshold_results = []

    for threshold in THRESHOLDS:

        metrics = calculate_metrics(
            y_adaptation,
            adaptation_meta_probability,
            threshold=float(threshold),
        )

        threshold_results.append(
            {
                "Threshold": float(
                    threshold
                ),
                **metrics,
            }
        )

    threshold_df = pd.DataFrame(
        threshold_results
    )

    # Primary objective: F1.
    # Secondary objective: recall.
    threshold_df = threshold_df.sort_values(
        by=[
            "F1-score",
            "Recall",
        ],
        ascending=False,
    ).reset_index(
        drop=True
    )

    selected_threshold = float(
        threshold_df.iloc[0][
            "Threshold"
        ]
    )

    threshold_df.to_csv(
        REPORT_DIR
        / "sh_das_v3_threshold_search.csv",
        index=False,
    )

    print(
        f"\nSelected threshold: "
        f"{selected_threshold:.2f}"
    )

    print(
        f"Adaptation F1: "
        f"{threshold_df.iloc[0]['F1-score']:.6f}"
    )

    print(
        f"Adaptation Recall: "
        f"{threshold_df.iloc[0]['Recall']:.6f}"
    )

    # -------------------------------------------------------------------------
    # Final predictions
    # -------------------------------------------------------------------------

    sh_das_eval_probability = (
        meta_model.predict_proba(
            evaluation_meta
        )[:, 1]
    )

    # -------------------------------------------------------------------------
    # Evaluate conventional models
    # -------------------------------------------------------------------------

    rows = []

    for model_name in base_models:

        probability = (
            pipelines[
                model_name
            ]
            .predict_proba(
                X_eval_drifted
            )[:, 1]
        )

        metrics = calculate_metrics(
            y_eval,
            probability,
            threshold=0.50,
        )

        rows.append(
            {
                "Model": model_name,
                **metrics,
                "Threshold": 0.50,
                "Drift_Aware": False,
            }
        )

    # -------------------------------------------------------------------------
    # Evaluate SH-DAS
    # -------------------------------------------------------------------------

    sh_das_metrics = calculate_metrics(
        y_eval,
        sh_das_eval_probability,
        threshold=selected_threshold,
    )

    rows.append(
        {
            "Model": "SH-DAS V3 (Proposed)",
            **sh_das_metrics,
            "Threshold": selected_threshold,
            "Drift_Aware": True,
        }
    )

    results_df = pd.DataFrame(
        rows
    )

    results_df.to_csv(
        REPORT_DIR
        / "six_model_v3_final_comparison.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Print final comparison
    # -------------------------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "FINAL SIX-MODEL COMPARISON — SH-DAS V3"
    )

    print(
        "=" * 80
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Comparison against best conventional metric
    # -------------------------------------------------------------------------

    conventional = results_df[
        results_df["Model"]
        != "SH-DAS V3 (Proposed)"
    ]

    proposed = results_df[
        results_df["Model"]
        == "SH-DAS V3 (Proposed)"
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
                "Best_Conventional": (
                    conventional_best
                ),
                "SH_DAS_V3": (
                    proposed_value
                ),
                "Delta": (
                    proposed_value
                    - conventional_best
                ),
            }
        )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_df.to_csv(
        REPORT_DIR
        / "sh_das_v3_vs_conventional.csv",
        index=False,
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "SH-DAS V3 DELTA AGAINST BEST CONVENTIONAL MODEL"
    )

    print(
        "=" * 80
    )

    print(
        comparison_df.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Meta-model coefficients
    # -------------------------------------------------------------------------

    classifier = (
        meta_model.named_steps[
            "classifier"
        ]
    )

    coefficient_df = pd.DataFrame(
        {
            "Meta_Feature": (
                adaptation_meta.columns
            ),
            "Coefficient": (
                classifier.coef_[0]
            ),
        }
    )

    coefficient_df[
        "Absolute_Coefficient"
    ] = coefficient_df[
        "Coefficient"
    ].abs()

    coefficient_df = (
        coefficient_df
        .sort_values(
            "Absolute_Coefficient",
            ascending=False,
        )
    )

    coefficient_df.to_csv(
        REPORT_DIR
        / "sh_das_v3_meta_model_coefficients.csv",
        index=False,
    )

    print(
        "\nTop meta-model features:"
    )

    print(
        coefficient_df.head(15).to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # ROC-AUC chart
    # -------------------------------------------------------------------------

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        results_df["Model"],
        results_df["ROC-AUC"],
    )

    plt.xlabel(
        "Model"
    )

    plt.ylabel(
        "ROC-AUC"
    )

    plt.title(
        "Five Conventional Models vs SH-DAS V3"
    )

    plt.xticks(
        rotation=30,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR
        / "six_model_v3_roc_auc.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # -------------------------------------------------------------------------
    # Summary JSON
    # -------------------------------------------------------------------------

    summary = {
        "model_name": "SH-DAS V3",
        "full_name": (
            "Self-Healing Drift-Aware Stacking"
        ),
        "base_models": list(
            base_models.keys()
        ),
        "meta_model": (
            "Logistic Regression"
        ),
        "drifted_features": (
            DRIFT_FEATURES
        ),
        "selected_threshold": (
            selected_threshold
        ),
        "adaptation_size": len(
            X_adaptation
        ),
        "evaluation_size": len(
            X_eval
        ),
        "meta_feature_count": int(
            adaptation_meta.shape[1]
        ),
        "evaluation_protocol": (
            "Base models trained on training "
            "data. Meta-model trained only on "
            "base-model predictions and drift-aware "
            "features from adaptation data. Final "
            "metrics calculated on a separate "
            "held-out drifted evaluation set."
        ),
    }

    with (
        REPORT_DIR
        / "sh_das_v3_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print(
        "\n" + "=" * 80
    )

    print(
        "SH-DAS V3 EXPERIMENT COMPLETED"
    )

    print(
        "=" * 80
    )

    print(
        f"\nReports saved to:\n{REPORT_DIR}"
    )


if __name__ == "__main__":
    main()