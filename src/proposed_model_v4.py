"""
SH-DAS V4
---------
Robust Self-Healing Drift-Aware Stacking

Key improvement over V3:
The meta-model is trained using out-of-fold predictions generated under
controlled drift. This reduces dependence on a single adaptation split.

Pipeline:

1. Split data into:
      training
      adaptation
      final evaluation

2. Use 5-fold OOF training data to construct drifted meta-training examples.

3. Train five conventional base models on full training data.

4. Generate adaptation and final-evaluation base predictions.

5. Train several candidate meta-models.

6. Select the meta-model using adaptation F1.

7. Select the operating threshold using adaptation F1.

8. Evaluate exactly once on the held-out final evaluation set.

This is an academic prototype.
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
    / "proposed_model_v4"
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

    X = df.drop(
        columns=["Churn"]
    )

    if y.isna().any():
        raise ValueError(
            "Unexpected target values in Churn."
        )

    return X, y.astype(int)


# =============================================================================
# PREPROCESSOR
# =============================================================================

def create_preprocessor(X):

    numeric = (
        X.select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    categorical = (
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
                numeric,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical,
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
# DRIFT REFERENCE
# =============================================================================

def create_drift_reference(X):

    reference = {}

    for feature in DRIFT_FEATURES:

        if feature not in X.columns:
            continue

        values = pd.to_numeric(
            X[feature],
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


def create_drift_features(
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
# TRAIN BASE MODEL
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
# CREATE META FEATURES
# =============================================================================

def create_meta_features(
    base_models,
    pipelines,
    X,
    drift_reference,
):

    meta = create_drift_features(
        X,
        drift_reference,
    )

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

        probability_name = (
            f"{safe_name}_prob"
        )

        meta[
            probability_name
        ] = probability

        for drift_column in (
            create_drift_features(
                X,
                drift_reference,
            ).columns
        ):

            interaction_name = (
                f"{safe_name}_x_"
                f"{drift_column}"
            )

            meta[
                interaction_name
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
    probability,
    threshold=0.50,
):

    prediction = (
        probability
        >= threshold
    ).astype(int)

    return {
        "Accuracy": accuracy_score(
            y_true,
            prediction,
        ),

        "Precision": precision_score(
            y_true,
            prediction,
            zero_division=0,
        ),

        "Recall": recall_score(
            y_true,
            prediction,
            zero_division=0,
        ),

        "F1-score": f1_score(
            y_true,
            prediction,
            zero_division=0,
        ),

        "ROC-AUC": roc_auc_score(
            y_true,
            probability,
        ),
    }


# =============================================================================
# OOF META DATASET
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
        "BUILDING OUT-OF-FOLD DRIFTED META-DATASET"
    )

    print(
        "=" * 80
    )

    base_model_definitions = (
        build_base_models()
    )

    splitter = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    meta_parts = []

    target_parts = []

    for fold_number, (
        fold_train_index,
        fold_valid_index,
    ) in enumerate(
        splitter.split(
            X_train,
            y_train,
        ),
        start=1,
    ):

        print(
            f"\nFold {fold_number}/{N_FOLDS}"
        )

        X_fold_train = (
            X_train.iloc[
                fold_train_index
            ]
        )

        y_fold_train = (
            y_train.iloc[
                fold_train_index
            ]
        )

        X_fold_valid = (
            X_train.iloc[
                fold_valid_index
            ]
        )

        y_fold_valid = (
            y_train.iloc[
                fold_valid_index
            ]
        )

        # Simulate the same kind of drift
        # that the final experiment will contain.
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
            base_model_definitions.items()
        ):

            pipeline = train_pipeline(
                model,
                X_fold_train,
                y_fold_train,
            )

            fold_pipelines[
                model_name
            ] = pipeline

        fold_meta = create_meta_features(
            base_model_definitions,
            fold_pipelines,
            X_fold_valid_drifted,
            drift_reference,
        )

        meta_parts.append(
            fold_meta
        )

        target_parts.append(
            y_fold_valid.reset_index(
                drop=True
            )
        )

    meta_dataset = pd.concat(
        meta_parts,
        axis=0,
        ignore_index=True,
    )

    meta_target = pd.concat(
        target_parts,
        axis=0,
        ignore_index=True,
    )

    meta_dataset = (
        meta_dataset.reset_index(
            drop=True
        )
    )

    print(
        "\nOOF meta dataset shape:",
        meta_dataset.shape,
    )

    return (
        meta_dataset,
        meta_target,
    )


# =============================================================================
# META MODELS
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
# FULL-TRAINING BASE MODELS
# =============================================================================

def train_full_base_models(
    X_train,
    y_train,
):

    models = build_base_models()

    pipelines = {}

    print(
        "\n"
        + "=" * 80
    )

    print(
        "TRAINING FULL BASE MODELS"
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

    return models, pipelines


# =============================================================================
# THRESHOLD SELECTION
# =============================================================================

def select_threshold(
    model,
    X_adaptation_meta,
    y_adaptation,
):

    probabilities = (
        model.predict_proba(
            X_adaptation_meta
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
# MAIN
# =============================================================================

def main():

    print("=" * 80)

    print(
        "SELF-HEALING ML — SH-DAS V4"
    )

    print(
        "ROBUST DRIFT-AWARE STACKING"
    )

    print("=" * 80)

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    X, y = load_dataset()

    print(
        f"\nDataset: {X.shape[0]} rows"
    )

    print(
        f"Features: {X.shape[1]}"
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
        f"\nTraining:   {len(X_train)}"
    )

    print(
        f"Adaptation: {len(X_adaptation)}"
    )

    print(
        f"Evaluation: {len(X_eval)}"
    )

    # -------------------------------------------------------------------------
    # OOF meta training
    # -------------------------------------------------------------------------

    (
        oof_meta,
        oof_target,
    ) = build_oof_meta_dataset(
        X_train,
        y_train,
    )

    oof_meta.to_csv(
        REPORT_DIR
        / "oof_meta_training_features.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "target": oof_target
        }
    ).to_csv(
        REPORT_DIR
        / "oof_meta_training_target.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Full base models
    # -------------------------------------------------------------------------

    base_models, base_pipelines = (
        train_full_base_models(
            X_train,
            y_train,
        )
    )

    # -------------------------------------------------------------------------
    # Drift reference
    # -------------------------------------------------------------------------

    drift_reference = (
        create_drift_reference(
            X_train
        )
    )

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

    # -------------------------------------------------------------------------
    # Meta features
    # -------------------------------------------------------------------------

    adaptation_meta = (
        create_meta_features(
            base_models,
            base_pipelines,
            X_adaptation_drifted,
            drift_reference,
        )
    )

    evaluation_meta = (
        create_meta_features(
            base_models,
            base_pipelines,
            X_eval_drifted,
            drift_reference,
        )
    )

    print(
        "\nMeta feature count:",
        adaptation_meta.shape[1],
    )

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
    # Train candidate meta models
    # -------------------------------------------------------------------------

    meta_models = (
        build_meta_models()
    )

    meta_results = []

    trained_meta_models = {}

    print(
        "\n"
        + "=" * 80
    )

    print(
        "TESTING META-MODEL CANDIDATES"
    )

    print(
        "=" * 80
    )

    for name, model in (
        meta_models.items()
    ):

        print(
            f"\nTraining {name}..."
        )

        model.fit(
            oof_meta,
            oof_target,
        )

        trained_meta_models[
            name
        ] = model

        adaptation_probability = (
            model.predict_proba(
                adaptation_meta
            )[:, 1]
        )

        adaptation_metrics = (
            calculate_metrics(
                y_adaptation,
                adaptation_probability,
                threshold=0.50,
            )
        )

        meta_results.append(
            {
                "Meta_Model": name,
                **adaptation_metrics,
            }
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

    meta_results_df = pd.DataFrame(
        meta_results
    )

    meta_results_df.to_csv(
        REPORT_DIR
        / "meta_model_candidates.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Select meta-model using adaptation data
    # -------------------------------------------------------------------------

    selected_meta_name = (
        meta_results_df
        .sort_values(
            by=[
                "F1-score",
                "Recall",
                "ROC-AUC",
            ],
            ascending=False,
        )
        .iloc[0][
            "Meta_Model"
        ]
    )

    selected_meta_model = (
        trained_meta_models[
            selected_meta_name
        ]
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "SELECTED META-MODEL"
    )

    print(
        "=" * 80
    )

    print(
        selected_meta_name
    )

    # -------------------------------------------------------------------------
    # Threshold
    # -------------------------------------------------------------------------

    selected_threshold, threshold_table = (
        select_threshold(
            selected_meta_model,
            adaptation_meta,
            y_adaptation,
        )
    )

    threshold_table.to_csv(
        REPORT_DIR
        / "sh_das_v4_threshold_search.csv",
        index=False,
    )

    print(
        f"\nSelected threshold: "
        f"{selected_threshold:.2f}"
    )

    selected_adaptation_metrics = (
        threshold_table.iloc[0]
    )

    print(
        f"Adaptation F1 at selected threshold: "
        f"{selected_adaptation_metrics['F1-score']:.6f}"
    )

    print(
        f"Adaptation Recall at selected threshold: "
        f"{selected_adaptation_metrics['Recall']:.6f}"
    )

    # -------------------------------------------------------------------------
    # Final conventional model evaluation
    # -------------------------------------------------------------------------

    results = []

    for model_name in base_models:

        probability = (
            base_pipelines[
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

        results.append(
            {
                "Model": model_name,
                **metrics,
                "Threshold": 0.50,
                "Drift_Aware": False,
            }
        )

    # -------------------------------------------------------------------------
    # SH-DAS V4 evaluation
    # -------------------------------------------------------------------------

    proposed_probability = (
        selected_meta_model
        .predict_proba(
            evaluation_meta
        )[:, 1]
    )

    proposed_metrics = (
        calculate_metrics(
            y_eval,
            proposed_probability,
            threshold=selected_threshold,
        )
    )

    results.append(
        {
            "Model": (
                "SH-DAS V4 (Proposed)"
            ),
            **proposed_metrics,
            "Threshold": selected_threshold,
            "Drift_Aware": True,
        }
    )

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        REPORT_DIR
        / "six_model_v4_final_comparison.csv",
        index=False,
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINAL SIX-MODEL COMPARISON — SH-DAS V4"
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
    # Delta
    # -------------------------------------------------------------------------

    conventional = results_df[
        results_df["Model"]
        != "SH-DAS V4 (Proposed)"
    ]

    proposed = results_df[
        results_df["Model"]
        == "SH-DAS V4 (Proposed)"
    ].iloc[0]

    comparison_rows = []

    for metric in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
    ]:

        best_conventional = (
            conventional[
                metric
            ].max()
        )

        proposed_value = (
            proposed[
                metric
            ]
        )

        comparison_rows.append(
            {
                "Metric": metric,
                "Best_Conventional": (
                    best_conventional
                ),
                "SH_DAS_V4": (
                    proposed_value
                ),
                "Delta": (
                    proposed_value
                    - best_conventional
                ),
            }
        )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_df.to_csv(
        REPORT_DIR
        / "sh_das_v4_vs_conventional.csv",
        index=False,
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "SH-DAS V4 DELTA AGAINST BEST CONVENTIONAL"
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
        "Five Conventional Models vs SH-DAS V4"
    )

    plt.xticks(
        rotation=30,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR
        / "six_model_v4_roc_auc.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    summary = {
        "model_name": "SH-DAS V4",
        "full_name": (
            "Robust Self-Healing Drift-Aware "
            "Stacking"
        ),
        "base_models": list(
            base_models.keys()
        ),
        "meta_models_tested": list(
            meta_models.keys()
        ),
        "selected_meta_model": (
            selected_meta_name
        ),
        "drifted_features": (
            DRIFT_FEATURES
        ),
        "selected_threshold": (
            selected_threshold
        ),
        "oof_folds": N_FOLDS,
        "evaluation_protocol": (
            "Base models were trained using "
            "training data. Meta-training used "
            "out-of-fold predictions generated "
            "under controlled drift. Meta-model "
            "and threshold were selected using "
            "adaptation data. Final metrics were "
            "computed on a separate held-out "
            "drifted evaluation set."
        ),
    }

    with (
        REPORT_DIR
        / "sh_das_v4_summary.json"
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
        "\n"
        + "=" * 80
    )

    print(
        "SH-DAS V4 COMPLETED"
    )

    print(
        "=" * 80
    )

    print(
        f"\nReports saved to:\n{REPORT_DIR}"
    )


if __name__ == "__main__":
    main()