"""
DRIFT RESILIENCE EXPERIMENT
---------------------------

Purpose
-------
Compare the five conventional models and the proposed SH-DAS V4 model
under:

    1. No drift
    2. Controlled feature drift

The experiment measures:

    Accuracy
    Precision
    Recall
    F1-score
    ROC-AUC

It also calculates:

    Performance Drop
    Performance Retention (%)

The SAME final evaluation records are used in both conditions.
Only the feature values are changed in the drift condition.

Therefore, the experiment isolates the effect of the simulated
feature distribution shift.

Proposed model:
    SH-DAS V4
    Self-Healing Drift-Aware Stacking

Base models:
    Logistic Regression
    Decision Tree
    Random Forest
    Gradient Boosting
    Support Vector Machine

The SH-DAS meta-model is trained using out-of-fold predictions
under controlled drift, then evaluated on both no-drift and drifted
versions of the held-out evaluation set.

This is an academic prototype, not a production deployment system.
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
    HistGradientBoostingClassifier,
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
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
)
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
    / "drift_resilience"
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

N_FOLDS = 5

EPSILON = 1e-6

DRIFT_FEATURES = [
    "tenure",
    "MonthlyCharges",
]

THRESHOLDS = np.arange(
    0.20,
    0.81,
    0.01,
)


# =============================================================================
# DATASET
# =============================================================================

def load_dataset():
    """Load and prepare the Telco churn dataset."""

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
            "Unexpected target values were found."
        )

    X = df.drop(
        columns=["Churn"]
    )

    return X, y.astype(int)


# =============================================================================
# PREPROCESSOR
# =============================================================================

def create_preprocessor(X):

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
# FIVE BASE MODELS
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

def apply_controlled_drift(X):

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
# DRIFT FEATURES
# =============================================================================

def create_drift_reference(X_train):

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
                max(
                    values.std(),
                    EPSILON,
                )
            ),
        }

    return reference


def create_drift_meta_features(
    X,
    reference,
):

    result = pd.DataFrame(
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

        z = (
            values - mean
        ) / std

        z = z.fillna(0.0)

        result[
            f"{feature}_drift_z"
        ] = z

        result[
            f"{feature}_drift_abs"
        ] = z.abs()

    return result


# =============================================================================
# TRAIN PIPELINE
# =============================================================================

def train_pipeline(
    model,
    X_train,
    y_train,
):

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(
                    X_train
                ),
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
# META FEATURE CONSTRUCTION
# =============================================================================

def create_meta_features(
    base_models,
    pipelines,
    X,
    drift_reference,
):

    drift_meta = (
        create_drift_meta_features(
            X,
            drift_reference,
        )
    )

    meta = drift_meta.copy()

    for model_name in base_models:

        probability = (
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

        probability_column = (
            f"{safe_name}_prob"
        )

        meta[
            probability_column
        ] = probability

        for drift_column in (
            drift_meta.columns
        ):

            interaction_column = (
                f"{safe_name}_x_"
                f"{drift_column}"
            )

            meta[
                interaction_column
            ] = (
                probability
                * meta[
                    drift_column
                ]
            )

    return meta


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
# OOF META TRAINING
# =============================================================================

def build_oof_meta_dataset(
    X_train,
    y_train,
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "BUILDING OOF DRIFTED META-DATASET"
    )

    print(
        "=" * 80
    )

    models = build_base_models()

    splitter = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    meta_parts = []
    target_parts = []

    for fold_number, (
        train_indices,
        valid_indices,
    ) in enumerate(
        splitter.split(
            X_train,
            y_train,
        ),
        start=1,
    ):

        print(
            f"Fold {fold_number}/{N_FOLDS}"
        )

        X_fold_train = X_train.iloc[
            train_indices
        ]

        y_fold_train = y_train.iloc[
            train_indices
        ]

        X_fold_valid = X_train.iloc[
            valid_indices
        ]

        y_fold_valid = y_train.iloc[
            valid_indices
        ]

        X_fold_valid_drifted = (
            apply_controlled_drift(
                X_fold_valid
            )
        )

        drift_reference = (
            create_drift_reference(
                X_fold_train
            )
        )

        fold_pipelines = {}

        for model_name, model in (
            models.items()
        ):

            fold_pipelines[
                model_name
            ] = train_pipeline(
                model,
                X_fold_train,
                y_fold_train,
            )

        fold_meta = create_meta_features(
            models,
            fold_pipelines,
            X_fold_valid_drifted,
            drift_reference,
        )

        meta_parts.append(
            fold_meta.reset_index(
                drop=True
            )
        )

        target_parts.append(
            y_fold_valid.reset_index(
                drop=True
            )
        )

    meta_dataset = pd.concat(
        meta_parts,
        ignore_index=True,
    )

    meta_target = pd.concat(
        target_parts,
        ignore_index=True,
    )

    return (
        meta_dataset,
        meta_target,
    )


# =============================================================================
# META-MODELS
# =============================================================================

def build_meta_models():

    return {
        "Meta Logistic Regression":
            Pipeline(
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
                        "model",
                        LogisticRegression(
                            max_iter=3000,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),

        "Meta Random Forest":
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        ),
                    ),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=300,
                            max_depth=4,
                            min_samples_leaf=8,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),

        "Meta HistGradientBoosting":
            Pipeline(
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
                        "model",
                        HistGradientBoostingClassifier(
                            max_iter=150,
                            learning_rate=0.05,
                            max_leaf_nodes=15,
                            l2_regularization=1.0,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
    }


# =============================================================================
# THRESHOLD SEARCH
# =============================================================================

def select_threshold(
    model,
    X_adaptation,
    y_adaptation,
):

    probabilities = (
        model.predict_proba(
            X_adaptation
        )[:, 1]
    )

    rows = []

    for threshold in THRESHOLDS:

        metrics = calculate_metrics(
            y_adaptation,
            probabilities,
            threshold=float(
                threshold
            ),
        )

        rows.append(
            {
                "Threshold": float(
                    threshold
                ),
                **metrics,
            }
        )

    table = pd.DataFrame(
        rows
    )

    table = table.sort_values(
        by=[
            "F1-score",
            "Recall",
        ],
        ascending=False,
    ).reset_index(
        drop=True
    )

    return (
        float(
            table.iloc[0][
                "Threshold"
            ]
        ),
        table,
    )


# =============================================================================
# CALCULATE PERFORMANCE TABLE
# =============================================================================

def create_condition_results(
    model_name,
    y_true,
    probability,
    threshold,
    condition,
):

    metrics = calculate_metrics(
        y_true,
        probability,
        threshold,
    )

    return {
        "Model": model_name,
        "Condition": condition,
        "Threshold": threshold,
        **metrics,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print(
        "SELF-HEALING ML — DRIFT RESILIENCE EXPERIMENT"
    )
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------------

    X, y = load_dataset()

    print(
        f"\nDataset: {len(X)} rows"
    )

    print(
        f"Features: {X.shape[1]}"
    )

    # -------------------------------------------------------------------------
    # Same split structure used by proposed model.
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
    # Create drifted copies.
    #
    # IMPORTANT:
    # The no-drift and drift conditions use exactly the same evaluation rows.
    # Only the selected feature values change.
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
        "\nDrift features:"
    )

    for feature in DRIFT_FEATURES:
        print(
            f"  {feature}"
        )

    # -------------------------------------------------------------------------
    # Train five conventional models.
    # -------------------------------------------------------------------------

    models = build_base_models()

    pipelines = {}

    print(
        "\n"
        + "=" * 80
    )

    print(
        "TRAINING FIVE CONVENTIONAL MODELS"
    )

    print(
        "=" * 80
    )

    for model_name, model in (
        models.items()
    ):

        print(
            f"\nTraining {model_name}..."
        )

        pipelines[
            model_name
        ] = train_pipeline(
            model,
            X_train,
            y_train,
        )

    # -------------------------------------------------------------------------
    # OOF data for SH-DAS.
    # -------------------------------------------------------------------------

    (
        oof_meta,
        oof_target,
    ) = build_oof_meta_dataset(
        X_train,
        y_train,
    )

    print(
        f"\nOOF meta dataset: "
        f"{oof_meta.shape}"
    )

    # -------------------------------------------------------------------------
    # Full training drift reference.
    # -------------------------------------------------------------------------

    drift_reference = (
        create_drift_reference(
            X_train
        )
    )

    # -------------------------------------------------------------------------
    # Adaptation meta data.
    # -------------------------------------------------------------------------

    adaptation_meta = (
        create_meta_features(
            models,
            pipelines,
            X_adaptation_drifted,
            drift_reference,
        )
    )

    # -------------------------------------------------------------------------
    # NO-DRIFT evaluation meta data.
    #
    # This is intentionally created from the unchanged evaluation data.
    # -------------------------------------------------------------------------

    evaluation_meta_no_drift = (
        create_meta_features(
            models,
            pipelines,
            X_eval,
            drift_reference,
        )
    )

    # -------------------------------------------------------------------------
    # DRIFT evaluation meta data.
    # -------------------------------------------------------------------------

    evaluation_meta_drift = (
        create_meta_features(
            models,
            pipelines,
            X_eval_drifted,
            drift_reference,
        )
    )

    # -------------------------------------------------------------------------
    # Train candidate meta-models.
    # -------------------------------------------------------------------------

    meta_models = (
        build_meta_models()
    )

    trained_meta_models = {}

    candidate_rows = []

    print(
        "\n"
        + "=" * 80
    )

    print(
        "SELECTING SH-DAS META-MODEL"
    )

    print(
        "=" * 80
    )

    for name, meta_model in (
        meta_models.items()
    ):

        print(
            f"\nTraining {name}..."
        )

        meta_model.fit(
            oof_meta,
            oof_target,
        )

        trained_meta_models[
            name
        ] = meta_model

        adaptation_probability = (
            meta_model.predict_proba(
                adaptation_meta
            )[:, 1]
        )

        metrics = calculate_metrics(
            y_adaptation,
            adaptation_probability,
            threshold=0.50,
        )

        candidate_rows.append(
            {
                "Meta_Model": name,
                **metrics,
            }
        )

        print(
            f"Adaptation F1: "
            f"{metrics['F1-score']:.6f}"
        )

        print(
            f"Adaptation Recall: "
            f"{metrics['Recall']:.6f}"
        )

        print(
            f"Adaptation ROC-AUC: "
            f"{metrics['ROC-AUC']:.6f}"
        )

    candidate_df = pd.DataFrame(
        candidate_rows
    )

    candidate_df = candidate_df.sort_values(
        by=[
            "F1-score",
            "Recall",
            "ROC-AUC",
        ],
        ascending=False,
    ).reset_index(
        drop=True
    )

    selected_meta_name = (
        candidate_df.iloc[0][
            "Meta_Model"
        ]
    )

    selected_meta_model = (
        trained_meta_models[
            selected_meta_name
        ]
    )

    candidate_df.to_csv(
        REPORT_DIR
        / "meta_model_candidates.csv",
        index=False,
    )

    print(
        "\nSelected meta-model:"
    )

    print(
        selected_meta_name
    )

    # -------------------------------------------------------------------------
    # Select threshold using adaptation data ONLY.
    # -------------------------------------------------------------------------

    (
        selected_threshold,
        threshold_table,
    ) = select_threshold(
        selected_meta_model,
        adaptation_meta,
        y_adaptation,
    )

    threshold_table.to_csv(
        REPORT_DIR
        / "sh_das_threshold_search.csv",
        index=False,
    )

    print(
        f"Selected threshold: "
        f"{selected_threshold:.2f}"
    )

    # -------------------------------------------------------------------------
    # SIX MODEL EVALUATION
    # -------------------------------------------------------------------------

    results = []

    # Conventional models.
    for model_name in models:

        pipeline = pipelines[
            model_name
        ]

        # No drift.
        no_drift_probability = (
            pipeline.predict_proba(
                X_eval
            )[:, 1]
        )

        # Drift.
        drift_probability = (
            pipeline.predict_proba(
                X_eval_drifted
            )[:, 1]
        )

        results.append(
            create_condition_results(
                model_name,
                y_eval,
                no_drift_probability,
                0.50,
                "No Drift",
            )
        )

        results.append(
            create_condition_results(
                model_name,
                y_eval,
                drift_probability,
                0.50,
                "Drift",
            )
        )

    # -------------------------------------------------------------------------
    # Proposed SH-DAS.
    # -------------------------------------------------------------------------

    sh_das_no_drift_probability = (
        selected_meta_model.predict_proba(
            evaluation_meta_no_drift
        )[:, 1]
    )

    sh_das_drift_probability = (
        selected_meta_model.predict_proba(
            evaluation_meta_drift
        )[:, 1]
    )

    results.append(
        create_condition_results(
            "SH-DAS V4 (Proposed)",
            y_eval,
            sh_das_no_drift_probability,
            selected_threshold,
            "No Drift",
        )
    )

    results.append(
        create_condition_results(
            "SH-DAS V4 (Proposed)",
            y_eval,
            sh_das_drift_probability,
            selected_threshold,
            "Drift",
        )
    )

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        REPORT_DIR
        / "no_drift_vs_drift_results.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Performance degradation and retention.
    # -------------------------------------------------------------------------

    comparison_rows = []

    metrics_to_compare = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
    ]

    for model_name in (
        results_df["Model"]
        .unique()
    ):

        model_results = results_df[
            results_df["Model"]
            == model_name
        ]

        no_drift = model_results[
            model_results["Condition"]
            == "No Drift"
        ].iloc[0]

        drift = model_results[
            model_results["Condition"]
            == "Drift"
        ].iloc[0]

        for metric in metrics_to_compare:

            no_drift_value = float(
                no_drift[metric]
            )

            drift_value = float(
                drift[metric]
            )

            performance_drop = (
                no_drift_value
                - drift_value
            )

            if abs(no_drift_value) <= EPSILON:
                retention = np.nan
            else:
                retention = (
                    drift_value
                    / no_drift_value
                    * 100.0
                )

            comparison_rows.append(
                {
                    "Model": model_name,
                    "Metric": metric,
                    "No_Drift": no_drift_value,
                    "Drift": drift_value,
                    "Performance_Drop": performance_drop,
                    "Retention_Percent": retention,
                }
            )

    degradation_df = pd.DataFrame(
        comparison_rows
    )

    degradation_df.to_csv(
        REPORT_DIR
        / "performance_degradation.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # ROC-AUC resilience summary.
    # -------------------------------------------------------------------------

    resilience_rows = []

    for model_name in (
        results_df["Model"]
        .unique()
    ):

        model_auc = degradation_df[
            (
                degradation_df["Model"]
                == model_name
            )
            &
            (
                degradation_df["Metric"]
                == "ROC-AUC"
            )
        ].iloc[0]

        resilience_rows.append(
            {
                "Model": model_name,
                "No_Drift_ROC_AUC": (
                    model_auc["No_Drift"]
                ),
                "Drift_ROC_AUC": (
                    model_auc["Drift"]
                ),
                "ROC_AUC_Drop": (
                    model_auc["Performance_Drop"]
                ),
                "ROC_AUC_Retention_Percent": (
                    model_auc["Retention_Percent"]
                ),
            }
        )

    resilience_df = pd.DataFrame(
        resilience_rows
    )

    resilience_df.to_csv(
        REPORT_DIR
        / "resilience_summary.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Print important output.
    # -------------------------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        "NO-DRIFT VS DRIFT RESULTS"
    )

    print(
        "=" * 80
    )

    display_columns = [
        "Model",
        "Condition",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "ROC-AUC DRIFT RESILIENCE"
    )

    print(
        "=" * 80
    )

    print(
        resilience_df.to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "DETAILED PERFORMANCE DEGRADATION"
    )

    print(
        "=" * 80
    )

    print(
        degradation_df.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # ROC-AUC resilience chart.
    # -------------------------------------------------------------------------

    chart_df = resilience_df.copy()

    x = np.arange(
        len(chart_df)
    )

    width = 0.35

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        x - width / 2,
        chart_df[
            "No_Drift_ROC_AUC"
        ],
        width,
        label="No Drift",
    )

    plt.bar(
        x + width / 2,
        chart_df[
            "Drift_ROC_AUC"
        ],
        width,
        label="Drift",
    )

    plt.xlabel(
        "Model"
    )

    plt.ylabel(
        "ROC-AUC"
    )

    plt.title(
        "Model Performance Before and After Simulated Drift"
    )

    plt.xticks(
        x,
        chart_df[
            "Model"
        ],
        rotation=30,
        ha="right",
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR
        / "drift_resilience_roc_auc.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # -------------------------------------------------------------------------
    # Retention chart.
    # -------------------------------------------------------------------------

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        chart_df[
            "Model"
        ],
        chart_df[
            "ROC_AUC_Retention_Percent"
        ],
    )

    plt.axhline(
        100.0,
        linewidth=1,
    )

    plt.xlabel(
        "Model"
    )

    plt.ylabel(
        "ROC-AUC Retention (%)"
    )

    plt.title(
        "ROC-AUC Retention Under Simulated Drift"
    )

    plt.xticks(
        rotation=30,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR
        / "roc_auc_retention.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # -------------------------------------------------------------------------
    # Summary JSON.
    # -------------------------------------------------------------------------

    summary = {
        "experiment": (
            "Drift Resilience Benchmark"
        ),
        "dataset": (
            "IBM Telco Customer Churn"
        ),
        "base_models": list(
            models.keys()
        ),
        "proposed_model": (
            "SH-DAS V4"
        ),
        "drift_features": (
            DRIFT_FEATURES
        ),
        "drift_description": {
            "tenure": "multiply by 1.5, cap at 72",
            "MonthlyCharges": (
                "multiply by 1.2, cap at 120"
            ),
        },
        "training_records": len(
            X_train
        ),
        "adaptation_records": len(
            X_adaptation
        ),
        "evaluation_records": len(
            X_eval
        ),
        "same_evaluation_rows": True,
        "selected_meta_model": (
            selected_meta_name
        ),
        "selected_threshold": (
            selected_threshold
        ),
        "interpretation": (
            "Performance retention measures the "
            "percentage of the no-drift metric "
            "retained under simulated drift. "
            "Higher retention indicates less "
            "relative degradation for that metric. "
            "This is an experiment-specific "
            "resilience measure."
        ),
    }

    with (
        REPORT_DIR
        / "drift_resilience_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Finish.
    # -------------------------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        "DRIFT RESILIENCE EXPERIMENT COMPLETED"
    )

    print(
        "=" * 80
    )

    print(
        f"\nReports saved to:\n{REPORT_DIR}"
    )


if __name__ == "__main__":
    main()