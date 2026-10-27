"""
SH-DAS V6
Performance-Optimized Drift-Aware Stacking

Project:
Self-Healing ML — Intelligent Framework for Detecting, Diagnosing
and Recovering from Model Drift.

V6 objectives
--------------
1. Train five conventional base classifiers.
2. Introduce controlled feature drift.
3. Construct drift-aware stacking meta-features.
4. Use 5-fold cross-validation on the adaptation set to evaluate
   multiple meta-model candidates.
5. Optimize the decision threshold using adaptation data only.
6. Select the V6 meta-model using:
      a) Cross-validated F1
      b) Cross-validated ROC-AUC
      c) Cross-validated Recall
7. Retrain the selected meta-model on all adaptation data.
8. Evaluate exactly once on the untouched evaluation set.
9. Save all experiment artifacts.

Important methodological rule
-----------------------------
The final evaluation set is NEVER used to:
- select the meta-model
- select hyperparameters
- select the threshold
- select the proposed architecture

This keeps the final comparison separated from model selection.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
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


# ============================================================================
# 1. PROJECT PATHS
# ============================================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

DATA_DIR = (
    PROJECT_ROOT / "data"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "proposed_model_v6"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================================
# 2. CONFIGURATION
# ============================================================================

RANDOM_STATE = 42

TRAIN_FRACTION = 0.60
ADAPTATION_FRACTION = 0.20
EVALUATION_FRACTION = 0.20

DRIFT_TRANSFORMS = {
    "tenure": 1.5,
    "MonthlyCharges": 1.2,
}

THRESHOLD_MIN = 0.20
THRESHOLD_MAX = 0.70
THRESHOLD_STEP = 0.01


# ============================================================================
# 3. DATASET DISCOVERY
# ============================================================================

def find_dataset() -> Path:
    """
    Return the actual IBM Telco Customer Churn dataset
    used by the project.
    """

    dataset_path = (
        DATA_DIR
        / "raw"
        / "Telco-Customer-Churn.csv"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    return dataset_path


# ============================================================================
# 4. METRIC EVALUATION
# ============================================================================

def evaluate_model(
    y_true,
    probabilities,
    threshold: float = 0.50,
) -> dict[str, float]:
    """
    Convert probabilities to predictions using the supplied
    threshold and calculate standard classification metrics.
    """

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "Accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "Precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "Recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "F1-score": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "ROC-AUC": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
    }


# ============================================================================
# 5. THRESHOLD SEARCH
# ============================================================================

def find_best_threshold(
    y_true,
    probabilities,
) -> float:
    """
    Select a threshold using adaptation data.

    Primary objective:
        Maximize F1-score.

    This threshold must never be optimized using the final
    evaluation data.
    """

    thresholds = np.arange(
        THRESHOLD_MIN,
        THRESHOLD_MAX + THRESHOLD_STEP / 2,
        THRESHOLD_STEP,
    )

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in thresholds:

        predictions = (
            np.asarray(probabilities)
            >= threshold
        ).astype(int)

        current_f1 = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        if current_f1 > best_f1:

            best_f1 = float(
                current_f1
            )

            best_threshold = float(
                round(threshold, 2)
            )

    return best_threshold


# ============================================================================
# 6. PREPROCESSOR
# ============================================================================

def build_preprocessor(
    X: pd.DataFrame,
) -> ColumnTransformer:
    """
    Build a common preprocessing pipeline.

    Numeric:
        median imputation + standardization

    Categorical:
        most-frequent imputation + one-hot encoding
    """

    numeric_features = (
        X.select_dtypes(
            include=[
                "int64",
                "float64",
                "int32",
                "float32",
            ]
        )
        .columns
        .tolist()
    )

    categorical_features = (
        X.select_dtypes(
            include=[
                "object",
                "category",
                "bool",
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
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    transformers = []

    if numeric_features:

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            )
        )

    if categorical_features:

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    return ColumnTransformer(
        transformers=transformers
    )


# ============================================================================
# 7. FIVE BASE MODELS
# ============================================================================

def build_base_models() -> dict:
    """
    Five heterogeneous conventional classifiers.
    """

    return {

        "Logistic Regression":
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                class_weight="balanced",
                max_depth=6,
                min_samples_leaf=10,
                random_state=RANDOM_STATE,
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=250,
                class_weight="balanced",
                min_samples_leaf=3,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),

        "Gradient Boosting":
            GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=3,
                random_state=RANDOM_STATE,
            ),

        "SVM":
            SVC(
                C=1.0,
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
    }


# ============================================================================
# 8. CONTROLLED DRIFT
# ============================================================================

def introduce_drift(
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Introduce controlled synthetic distribution drift.

    tenure:
        multiply by 1.5, capped at 72

    MonthlyCharges:
        multiply by 1.2, capped at 120
    """

    X_drifted = X.copy()

    if "tenure" in X_drifted.columns:

        X_drifted["tenure"] = (
            pd.to_numeric(
                X_drifted["tenure"],
                errors="coerce",
            )
            * DRIFT_TRANSFORMS["tenure"]
        )

        X_drifted["tenure"] = (
            X_drifted["tenure"]
            .clip(upper=72)
        )

    if "MonthlyCharges" in X_drifted.columns:

        X_drifted["MonthlyCharges"] = (
            pd.to_numeric(
                X_drifted["MonthlyCharges"],
                errors="coerce",
            )
            * DRIFT_TRANSFORMS[
                "MonthlyCharges"
            ]
        )

        X_drifted["MonthlyCharges"] = (
            X_drifted["MonthlyCharges"]
            .clip(upper=120)
        )

    return X_drifted


# ============================================================================
# 9. NUMERIC FEATURE DRIFT
# ============================================================================

def compute_drift_features(
    reference: pd.DataFrame,
    current: pd.DataFrame,
):
    """
    Calculate feature-level distribution indicators.

    For each numeric feature:
        absolute mean shift
        standardized mean shift

    The values describe the difference between the reference
    training distribution and the current drifted distribution.
    """

    reference_numeric = (
        reference.select_dtypes(
            include=[
                "int64",
                "float64",
                "int32",
                "float32",
            ]
        )
    )

    current_numeric = (
        current.select_dtypes(
            include=[
                "int64",
                "float64",
                "int32",
                "float32",
            ]
        )
    )

    shared_features = [
        feature
        for feature in reference_numeric.columns
        if feature in current_numeric.columns
    ]

    drift_abs = {}
    drift_z = {}

    for feature in shared_features:

        ref_values = pd.to_numeric(
            reference_numeric[feature],
            errors="coerce",
        ).dropna()

        cur_values = pd.to_numeric(
            current_numeric[feature],
            errors="coerce",
        ).dropna()

        if ref_values.empty:

            drift_abs[feature] = 0.0
            drift_z[feature] = 0.0

            continue

        ref_mean = float(
            ref_values.mean()
        )

        ref_std = float(
            ref_values.std()
        )

        if cur_values.empty:

            current_mean = ref_mean

        else:

            current_mean = float(
                cur_values.mean()
            )

        mean_shift = (
            current_mean
            - ref_mean
        )

        drift_abs[feature] = float(
            abs(mean_shift)
        )

        if (
            ref_std == 0
            or np.isnan(ref_std)
        ):

            drift_z[feature] = 0.0

        else:

            drift_z[feature] = float(
                mean_shift / ref_std
            )

    return (
        drift_abs,
        drift_z,
    )


# ============================================================================
# 10. TRAIN BASE MODELS AND GENERATE PROBABILITIES
# ============================================================================

def train_base_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_target: pd.DataFrame,
):
    """
    Train the five base classifiers and produce:
        probabilities
        binary predictions
    """

    trained_models = {}

    probability_dict = {}

    prediction_dict = {}

    models = build_base_models()

    for name, estimator in models.items():

        print(
            f"Training {name}..."
        )

        preprocessor = build_preprocessor(
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
                    estimator,
                ),
            ]
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        probabilities = (
            pipeline
            .predict_proba(
                X_target
            )[:, 1]
        )

        predictions = (
            probabilities >= 0.50
        ).astype(int)

        trained_models[name] = (
            pipeline
        )

        probability_dict[name] = (
            probabilities
        )

        prediction_dict[name] = (
            predictions
        )

    return (
        trained_models,
        probability_dict,
        prediction_dict,
    )


# ============================================================================
# 11. META-FEATURE CONSTRUCTION
# ============================================================================

def build_meta_features(
    probabilities: dict,
    predictions: dict,
    drift_abs: dict,
    drift_z: dict,
):
    """
    Construct drift-aware stacking features.

    Components:
        1. Five base-model probability outputs.
        2. Five base-model binary predictions.
        3. Feature-level absolute drift values.
        4. Feature-level standardized drift values.
        5. Global drift summaries.
        6. Model × drift interaction terms.
    """

    first_probability = (
        next(
            iter(probabilities.values())
        )
    )

    number_of_rows = len(
        first_probability
    )

    meta_df = pd.DataFrame(
        index=range(
            number_of_rows
        )
    )

    # ------------------------------------------------------------------------
    # Base probabilities
    # ------------------------------------------------------------------------

    probability_columns = []

    for model_name, values in (
        probabilities.items()
    ):

        safe_name = (
            model_name
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        column_name = (
            f"{safe_name}_prob"
        )

        meta_df[column_name] = (
            values
        )

        probability_columns.append(
            column_name
        )

    # ------------------------------------------------------------------------
    # Base predictions
    # ------------------------------------------------------------------------

    for model_name, values in (
        predictions.items()
    ):

        safe_name = (
            model_name
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        column_name = (
            f"{safe_name}_pred"
        )

        meta_df[column_name] = (
            values
        )

    # ------------------------------------------------------------------------
    # Feature drift
    # ------------------------------------------------------------------------

    drift_columns = []

    for feature, value in (
        drift_abs.items()
    ):

        column_name = (
            f"{feature}_drift_abs"
        )

        meta_df[column_name] = (
            float(value)
        )

        drift_columns.append(
            column_name
        )

    for feature, value in (
        drift_z.items()
    ):

        column_name = (
            f"{feature}_drift_z"
        )

        meta_df[column_name] = (
            float(value)
        )

        drift_columns.append(
            column_name
        )

    # ------------------------------------------------------------------------
    # Global drift summaries
    # ------------------------------------------------------------------------

    abs_values = list(
        drift_abs.values()
    )

    z_values = list(
        drift_z.values()
    )

    if abs_values:

        meta_df[
            "global_drift_abs_mean"
        ] = float(
            np.mean(abs_values)
        )

        meta_df[
            "global_drift_abs_max"
        ] = float(
            np.max(abs_values)
        )

    else:

        meta_df[
            "global_drift_abs_mean"
        ] = 0.0

        meta_df[
            "global_drift_abs_max"
        ] = 0.0

    if z_values:

        meta_df[
            "global_drift_z_mean"
        ] = float(
            np.mean(
                np.abs(z_values)
            )
        )

        meta_df[
            "global_drift_z_max"
        ] = float(
            np.max(
                np.abs(z_values)
            )
        )

    else:

        meta_df[
            "global_drift_z_mean"
        ] = 0.0

        meta_df[
            "global_drift_z_max"
        ] = 0.0

    # ------------------------------------------------------------------------
    # Model × drift interaction terms
    # ------------------------------------------------------------------------

    for probability_column in (
        probability_columns
    ):

        model_prefix = (
            probability_column
            .replace(
                "_prob",
                "",
            )
        )

        for drift_column in (
            drift_columns
        ):

            interaction_column = (
                f"{model_prefix}_x_"
                f"{drift_column}"
            )

            meta_df[
                interaction_column
            ] = (
                meta_df[
                    probability_column
                ]
                * meta_df[
                    drift_column
                ]
            )

    return meta_df


# ============================================================================
# 12. V6 META-MODEL CANDIDATES
# ============================================================================

def build_meta_candidates() -> dict:
    """
    Candidate meta-model search space.

    Logistic Regression:
        several regularization values
        balanced and unbalanced class weighting

    Random Forest:
        standard and conservative versions

    HistGradientBoosting:
        standard and conservative versions
    """

    candidates = {}

    # ------------------------------------------------------------------------
    # Logistic Regression candidates
    # ------------------------------------------------------------------------

    logistic_settings = [
        (0.03, "balanced"),
        (0.10, "balanced"),
        (0.30, "balanced"),
        (1.00, "balanced"),
        (3.00, "balanced"),
        (10.0, "balanced"),
        (0.03, None),
        (0.10, None),
        (0.30, None),
        (1.00, None),
        (3.00, None),
        (10.0, None),
    ]

    for C_value, class_weight in (
        logistic_settings
    ):

        weight_label = (
            "balanced"
            if class_weight == "balanced"
            else "none"
        )

        candidate_name = (
            "Meta Logistic "
            f"C={C_value} "
            f"weight={weight_label}"
        )

        candidates[
            candidate_name
        ] = Pipeline(
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
                        C=C_value,
                        max_iter=3000,
                        class_weight=class_weight,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

    # ------------------------------------------------------------------------
    # Random Forest
    # ------------------------------------------------------------------------

    candidates[
        "Meta Random Forest"
    ] = Pipeline(
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
                    max_depth=5,
                    min_samples_leaf=8,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    candidates[
        "Meta Random Forest Conservative"
    ] = Pipeline(
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
                    min_samples_leaf=12,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # ------------------------------------------------------------------------
    # HistGradientBoosting
    # ------------------------------------------------------------------------

    candidates[
        "Meta HistGradientBoosting"
    ] = Pipeline(
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
                    max_leaf_nodes=12,
                    l2_regularization=1.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    candidates[
        "Meta HistGradientBoosting Conservative"
    ] = Pipeline(
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
                    max_iter=120,
                    learning_rate=0.03,
                    max_leaf_nodes=8,
                    l2_regularization=2.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return candidates


# ============================================================================
# 13. CROSS-VALIDATED CANDIDATE EVALUATION
# ============================================================================

def evaluate_candidate_with_cv(
    candidate_model,
    X_meta: pd.DataFrame,
    y_meta: pd.Series,
):
    """
    Evaluate one meta-model using 5-fold stratified
    cross-validation on the adaptation data.

    Returns:
        overall OOF metrics
        selected threshold
        per-fold metrics
        OOF probabilities

    The final evaluation set is not touched.
    """

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof_probabilities = np.zeros(
        len(y_meta),
        dtype=float,
    )

    fold_rows = []

    for fold_number, (
        train_indices,
        valid_indices,
    ) in enumerate(
        cv.split(
            X_meta,
            y_meta,
        ),
        start=1,
    ):

        print(
            f"    CV Fold "
            f"{fold_number}/5"
        )

        fold_model = clone(
            candidate_model
        )

        X_fold_train = (
            X_meta.iloc[
                train_indices
            ]
        )

        X_fold_valid = (
            X_meta.iloc[
                valid_indices
            ]
        )

        y_fold_train = (
            y_meta.iloc[
                train_indices
            ]
        )

        y_fold_valid = (
            y_meta.iloc[
                valid_indices
            ]
        )

        fold_model.fit(
            X_fold_train,
            y_fold_train,
        )

        valid_probabilities = (
            fold_model
            .predict_proba(
                X_fold_valid
            )[:, 1]
        )

        oof_probabilities[
            valid_indices
        ] = valid_probabilities

        fold_threshold = (
            find_best_threshold(
                y_fold_valid,
                valid_probabilities,
            )
        )

        fold_metrics = evaluate_model(
            y_fold_valid,
            valid_probabilities,
            fold_threshold,
        )

        fold_rows.append(
            {
                "Fold": fold_number,
                "Threshold":
                    fold_threshold,
                "Accuracy":
                    fold_metrics[
                        "Accuracy"
                    ],
                "Precision":
                    fold_metrics[
                        "Precision"
                    ],
                "Recall":
                    fold_metrics[
                        "Recall"
                    ],
                "F1-score":
                    fold_metrics[
                        "F1-score"
                    ],
                "ROC-AUC":
                    fold_metrics[
                        "ROC-AUC"
                    ],
            }
        )

    # ------------------------------------------------------------------------
    # Global OOF threshold
    # ------------------------------------------------------------------------

    selected_threshold = (
        find_best_threshold(
            y_meta,
            oof_probabilities,
        )
    )

    overall_metrics = evaluate_model(
        y_meta,
        oof_probabilities,
        selected_threshold,
    )

    fold_df = pd.DataFrame(
        fold_rows
    )

    return (
        overall_metrics,
        selected_threshold,
        fold_df,
        oof_probabilities,
    )


# ============================================================================
# 14. MAIN EXPERIMENT
# ============================================================================

def main():

    print()
    print("=" * 75)
    print(
        "SH-DAS V6 — "
        "CROSS-VALIDATED PERFORMANCE-OPTIMIZED "
        "DRIFT-AWARE STACKING"
    )
    print("=" * 75)

    # ========================================================================
    # DATASET
    # ========================================================================

    dataset_path = find_dataset()

    print()
    print(
        f"Dataset: {dataset_path}"
    )

    df = pd.read_csv(
        dataset_path
    )

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    if "customerID" in df.columns:

        df = df.drop(
            columns=[
                "customerID"
            ]
        )

    if "Churn" not in df.columns:

        raise ValueError(
            "Dataset must contain the "
            "'Churn' column."
        )

    # ========================================================================
    # TARGET
    # ========================================================================

    y = (
        df["Churn"]
        .astype(str)
        .str.strip()
        .map(
            {
                "Yes": 1,
                "No": 0,
            }
        )
    )

    if y.isna().any():

        raise ValueError(
            "Unexpected values found in "
            "the Churn target column."
        )

    y = y.astype(int)

    # ========================================================================
    # FEATURES
    # ========================================================================

    X = df.drop(
        columns=[
            "Churn"
        ]
    ).copy()

    if "TotalCharges" in X.columns:

        X["TotalCharges"] = (
            pd.to_numeric(
                X["TotalCharges"],
                errors="coerce",
            )
        )

    print(
        f"Dataset shape: {X.shape}"
    )

    # ========================================================================
    # THREE-WAY SPLIT
    # ========================================================================

    (
        X_temp,
        X_eval,
        y_temp,
        y_eval,
    ) = train_test_split(
        X,
        y,
        test_size=EVALUATION_FRACTION,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    adaptation_fraction_of_temp = (
        ADAPTATION_FRACTION
        / (
            TRAIN_FRACTION
            + ADAPTATION_FRACTION
        )
    )

    (
        X_train,
        X_adaptation,
        y_train,
        y_adaptation,
    ) = train_test_split(
        X_temp,
        y_temp,
        test_size=adaptation_fraction_of_temp,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    print()
    print(
        f"Training records:   "
        f"{len(X_train)}"
    )

    print(
        f"Adaptation records: "
        f"{len(X_adaptation)}"
    )

    print(
        f"Evaluation records: "
        f"{len(X_eval)}"
    )

    # ========================================================================
    # CONTROLLED DRIFT
    # ========================================================================

    X_adaptation_drift = (
        introduce_drift(
            X_adaptation
        )
    )

    X_eval_drift = (
        introduce_drift(
            X_eval
        )
    )

    print()
    print(
        "Controlled drift introduced:"
    )

    print(
        "  tenure × 1.5, capped at 72"
    )

    print(
        "  MonthlyCharges × 1.2, "
        "capped at 120"
    )

    # ========================================================================
    # TRAIN BASE MODELS ON ADAPTATION DATA
    # ========================================================================

    print()
    print("=" * 75)
    print(
        "TRAINING BASE MODELS"
    )
    print("=" * 75)

    (
        base_models_adaptation,
        adaptation_probabilities,
        adaptation_predictions,
    ) = train_base_models(
        X_train,
        y_train,
        X_adaptation_drift,
    )

    # ========================================================================
    # FINAL EVALUATION BASE-MODEL PREDICTIONS
    # ========================================================================

    (
        base_models_evaluation,
        evaluation_probabilities,
        evaluation_predictions,
    ) = train_base_models(
        X_train,
        y_train,
        X_eval_drift,
    )

    # ========================================================================
    # DRIFT FEATURES
    # ========================================================================

    (
        adaptation_drift_abs,
        adaptation_drift_z,
    ) = compute_drift_features(
        X_train,
        X_adaptation_drift,
    )

    (
        evaluation_drift_abs,
        evaluation_drift_z,
    ) = compute_drift_features(
        X_train,
        X_eval_drift,
    )

    # ========================================================================
    # BUILD META-FEATURES
    # ========================================================================

    meta_adaptation = (
        build_meta_features(
            adaptation_probabilities,
            adaptation_predictions,
            adaptation_drift_abs,
            adaptation_drift_z,
        )
    )

    meta_evaluation = (
        build_meta_features(
            evaluation_probabilities,
            evaluation_predictions,
            evaluation_drift_abs,
            evaluation_drift_z,
        )
    )

    print()
    print(
        f"Meta-feature count: "
        f"{meta_adaptation.shape[1]}"
    )

    # ========================================================================
    # SAVE META DATA
    # ========================================================================

    meta_adaptation.to_csv(
        REPORT_DIR
        / "adaptation_meta_features.csv",
        index=False,
    )

    meta_evaluation.to_csv(
        REPORT_DIR
        / "evaluation_meta_features.csv",
        index=False,
    )

    # ========================================================================
    # V6 META-MODEL SEARCH
    # ========================================================================

    print()
    print("=" * 75)
    print(
        "V6 CROSS-VALIDATED META-MODEL SEARCH"
    )
    print("=" * 75)

    candidate_rows = []

    candidate_fold_tables = {}

    candidates = (
        build_meta_candidates()
    )

    for (
        candidate_name,
        candidate_model,
    ) in candidates.items():

        print()
        print(
            f"Evaluating "
            f"{candidate_name}..."
        )

        start_time = (
            time.perf_counter()
        )

        (
            metrics,
            threshold,
            fold_df,
            oof_probabilities,
        ) = evaluate_candidate_with_cv(
            candidate_model,
            meta_adaptation,
            y_adaptation,
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        candidate_rows.append(
            {
                "Candidate":
                    candidate_name,

                "Threshold":
                    threshold,

                "Accuracy":
                    metrics[
                        "Accuracy"
                    ],

                "Precision":
                    metrics[
                        "Precision"
                    ],

                "Recall":
                    metrics[
                        "Recall"
                    ],

                "F1-score":
                    metrics[
                        "F1-score"
                    ],

                "ROC-AUC":
                    metrics[
                        "ROC-AUC"
                    ],

                "CV_F1_Mean":
                    fold_df[
                        "F1-score"
                    ].mean(),

                "CV_F1_STD":
                    fold_df[
                        "F1-score"
                    ].std(),

                "CV_AUC_Mean":
                    fold_df[
                        "ROC-AUC"
                    ].mean(),

                "CV_Recall_Mean":
                    fold_df[
                        "Recall"
                    ].mean(),

                "Time_sec":
                    elapsed,
            }
        )

        safe_candidate_name = (
            candidate_name
            .lower()
            .replace(" ", "_")
            .replace("=", "")
            .replace(".", "_")
        )

        fold_df.to_csv(
            REPORT_DIR
            / (
                "cv_folds_"
                f"{safe_candidate_name}.csv"
            ),
            index=False,
        )

        candidate_fold_tables[
            candidate_name
        ] = fold_df

    # ========================================================================
    # RANK META-MODEL CANDIDATES
    # ========================================================================

    candidates_df = pd.DataFrame(
        candidate_rows
    )

    candidates_df = (
        candidates_df
        .sort_values(
            by=[
                "CV_F1_Mean",
                "CV_AUC_Mean",
                "CV_Recall_Mean",
            ],
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    candidates_df.to_csv(
        REPORT_DIR
        / "v6_meta_candidate_search.csv",
        index=False,
    )

    print()
    print(
        candidates_df.to_string(
            index=False
        )
    )

    # ========================================================================
    # SELECT BEST V6 CANDIDATE
    # ========================================================================

    selected_row = (
        candidates_df.iloc[0]
    )

    selected_candidate = str(
        selected_row[
            "Candidate"
        ]
    )

    selected_threshold = float(
        selected_row[
            "Threshold"
        ]
    )

    print()
    print("=" * 75)
    print(
        "SELECTED V6 CANDIDATE"
    )
    print("=" * 75)

    print(
        f"Candidate: "
        f"{selected_candidate}"
    )

    print(
        f"Threshold: "
        f"{selected_threshold:.2f}"
    )

    print(
        f"CV F1 mean: "
        f"{selected_row['CV_F1_Mean']:.4f}"
    )

    print(
        f"CV ROC-AUC mean: "
        f"{selected_row['CV_AUC_Mean']:.4f}"
    )

    print(
        f"CV Recall mean: "
        f"{selected_row['CV_Recall_Mean']:.4f}"
    )

    # ========================================================================
    # FIT SELECTED META-MODEL ON ALL ADAPTATION DATA
    # ========================================================================

    selected_model = clone(
        candidates[
            selected_candidate
        ]
    )

    selected_model.fit(
        meta_adaptation,
        y_adaptation,
    )

    # ========================================================================
    # FINAL PROPOSED MODEL EVALUATION
    # ========================================================================

    print()
    print("=" * 75)
    print(
        "SH-DAS V6 FINAL EVALUATION"
    )
    print("=" * 75)

    final_probability = (
        selected_model
        .predict_proba(
            meta_evaluation
        )[:, 1]
    )

    v6_metrics = (
        evaluate_model(
            y_eval,
            final_probability,
            selected_threshold,
        )
    )

    print(
        v6_metrics
    )

    # ========================================================================
    # CONVENTIONAL MODEL FINAL EVALUATION
    #
    # Thresholds are selected from adaptation data only.
    # ========================================================================

    comparison_rows = []

    for (
        model_name,
        probabilities,
    ) in evaluation_probabilities.items():

        adaptation_threshold = (
            find_best_threshold(
                y_adaptation,
                adaptation_probabilities[
                    model_name
                ],
            )
        )

        conventional_metrics = (
            evaluate_model(
                y_eval,
                probabilities,
                adaptation_threshold,
            )
        )

        comparison_rows.append(
            {
                "Model":
                    model_name,

                "Type":
                    "Conventional",

                "Threshold":
                    adaptation_threshold,

                "Accuracy":
                    conventional_metrics[
                        "Accuracy"
                    ],

                "Precision":
                    conventional_metrics[
                        "Precision"
                    ],

                "Recall":
                    conventional_metrics[
                        "Recall"
                    ],

                "F1-score":
                    conventional_metrics[
                        "F1-score"
                    ],

                "ROC-AUC":
                    conventional_metrics[
                        "ROC-AUC"
                    ],
            }
        )

    # ========================================================================
    # ADD SH-DAS V6
    # ========================================================================

    comparison_rows.append(
        {
            "Model":
                "SH-DAS V6",

            "Type":
                "Proposed",

            "Threshold":
                selected_threshold,

            "Accuracy":
                v6_metrics[
                    "Accuracy"
                ],

            "Precision":
                v6_metrics[
                    "Precision"
                ],

            "Recall":
                v6_metrics[
                    "Recall"
                ],

            "F1-score":
                v6_metrics[
                    "F1-score"
                ],

            "ROC-AUC":
                v6_metrics[
                    "ROC-AUC"
                ],
        }
    )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_df.to_csv(
        REPORT_DIR
        / "six_model_v6_final_comparison.csv",
        index=False,
    )

    # ========================================================================
    # FINAL PRINT
    # ========================================================================

    print()
    print("=" * 75)
    print(
        "FINAL MODEL COMPARISON"
    )
    print("=" * 75)

    print(
        comparison_df.to_string(
            index=False
        )
    )

    # ========================================================================
    # FIND BEST CONVENTIONAL VALUE FOR EACH METRIC
    # ========================================================================

    conventional_df = (
        comparison_df[
            comparison_df["Type"]
            == "Conventional"
        ]
        .copy()
    )

    delta_rows = []

    for metric in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
    ]:

        best_conventional = float(
            conventional_df[
                metric
            ].max()
        )

        v6_value = float(
            v6_metrics[
                metric
            ]
        )

        delta_rows.append(
            {
                "Metric":
                    metric,

                "Best_Conventional":
                    best_conventional,

                "SH_DAS_V6":
                    v6_value,

                "V6_minus_Best_Conventional":
                    v6_value
                    - best_conventional,
            }
        )

    delta_df = pd.DataFrame(
        delta_rows
    )

    delta_df.to_csv(
        REPORT_DIR
        / "sh_das_v6_vs_conventional.csv",
        index=False,
    )

    # ========================================================================
    # BASE MODEL METRICS
    # ========================================================================

    base_model_rows = []

    for (
        model_name,
        probabilities,
    ) in evaluation_probabilities.items():

        threshold = (
            find_best_threshold(
                y_adaptation,
                adaptation_probabilities[
                    model_name
                ],
            )
        )

        metrics = evaluate_model(
            y_eval,
            probabilities,
            threshold,
        )

        base_model_rows.append(
            {
                "Model":
                    model_name,

                "Threshold":
                    threshold,

                "Accuracy":
                    metrics[
                        "Accuracy"
                    ],

                "Precision":
                    metrics[
                        "Precision"
                    ],

                "Recall":
                    metrics[
                        "Recall"
                    ],

                "F1-score":
                    metrics[
                        "F1-score"
                    ],

                "ROC-AUC":
                    metrics[
                        "ROC-AUC"
                    ],
            }
        )

    pd.DataFrame(
        base_model_rows
    ).to_csv(
        REPORT_DIR
        / "v6_base_model_results.csv",
        index=False,
    )

    # ========================================================================
    # SAVE SELECTED MODEL
    # ========================================================================

    try:

        import joblib

        joblib.dump(
            selected_model,
            REPORT_DIR
            / "sh_das_v6_meta_model.joblib",
        )

    except ImportError:

        print(
            "joblib is not available; "
            "meta-model file was not saved."
        )

    # ========================================================================
    # SAVE PROBABILITIES
    # ========================================================================

    probability_output = pd.DataFrame(
        {
            "y_true": y_eval.values,

            "SH_DAS_V6_probability":
                final_probability,
        }
    )

    probability_output.to_csv(
        REPORT_DIR
        / "sh_das_v6_evaluation_probabilities.csv",
        index=False,
    )

    # ========================================================================
    # SAVE SUMMARY
    # ========================================================================

    summary = {

        "model_name":
            "SH-DAS V6",

        "dataset":
            str(dataset_path),

        "dataset_records":
            int(len(X)),

        "feature_count":
            int(X.shape[1]),

        "training_records":
            int(len(X_train)),

        "adaptation_records":
            int(len(X_adaptation)),

        "evaluation_records":
            int(len(X_eval)),

        "meta_feature_count":
            int(meta_adaptation.shape[1]),

        "selected_candidate":
            selected_candidate,

        "selected_threshold":
            selected_threshold,

        "cv_f1_mean":
            float(
                selected_row[
                    "CV_F1_Mean"
                ]
            ),

        "cv_auc_mean":
            float(
                selected_row[
                    "CV_AUC_Mean"
                ]
            ),

        "cv_recall_mean":
            float(
                selected_row[
                    "CV_Recall_Mean"
                ]
            ),

        "final_metrics": {
            key: float(value)
            for key, value
            in v6_metrics.items()
        },

        "metric_deltas": {
            row["Metric"]:
                float(
                    row[
                        "V6_minus_Best_Conventional"
                    ]
                )
            for row
            in delta_rows
        },

        "drift_transforms":
            DRIFT_TRANSFORMS,

        "selection_rule": [
            "5-fold stratified cross-validation "
            "on adaptation data",
            "Primary objective: mean adaptation F1-score",
            "Tie-break: mean adaptation ROC-AUC",
            "Final tie-break: mean adaptation Recall",
            "Threshold optimized only using adaptation data",
            "Final evaluation set kept untouched",
        ],
    }

    with (
        REPORT_DIR
        / "sh_das_v6_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    # ========================================================================
    # SAVE DRIFT SUMMARY
    # ========================================================================

    drift_rows = []

    all_features = sorted(
        set(
            adaptation_drift_abs.keys()
        )
        |
        set(
            adaptation_drift_z.keys()
        )
    )

    for feature in all_features:

        drift_rows.append(
            {
                "Feature":
                    feature,

                "Drift_Absolute_Mean_Shift":
                    adaptation_drift_abs.get(
                        feature,
                        0.0,
                    ),

                "Drift_Z":
                    adaptation_drift_z.get(
                        feature,
                        0.0,
                    ),
            }
        )

    pd.DataFrame(
        drift_rows
    ).sort_values(
        by="Drift_Absolute_Mean_Shift",
        ascending=False,
    ).to_csv(
        REPORT_DIR
        / "v6_feature_drift_summary.csv",
        index=False,
    )

    # ========================================================================
    # SAVE FINAL RESULT REPORT
    # ========================================================================

    report_lines = []

    report_lines.append(
        "SH-DAS V6 — FINAL REPORT"
    )

    report_lines.append(
        "=" * 75
    )

    report_lines.append(
        ""
    )

    report_lines.append(
        f"Dataset: {dataset_path}"
    )

    report_lines.append(
        f"Dataset records: {len(X)}"
    )

    report_lines.append(
        f"Features: {X.shape[1]}"
    )

    report_lines.append(
        f"Training records: {len(X_train)}"
    )

    report_lines.append(
        f"Adaptation records: {len(X_adaptation)}"
    )

    report_lines.append(
        f"Evaluation records: {len(X_eval)}"
    )

    report_lines.append(
        ""
    )

    report_lines.append(
        "Selected candidate:"
    )

    report_lines.append(
        f"  {selected_candidate}"
    )

    report_lines.append(
        f"Threshold: {selected_threshold:.2f}"
    )

    report_lines.append(
        ""
    )

    report_lines.append(
        "Final SH-DAS V6 metrics:"
    )

    for metric, value in (
        v6_metrics.items()
    ):

        report_lines.append(
            f"  {metric}: {value:.6f}"
        )

    report_lines.append(
        ""
    )

    report_lines.append(
        "Comparison:"
    )

    report_lines.append(
        comparison_df.to_string(
            index=False
        )
    )

    report_lines.append(
        ""
    )

    report_lines.append(
        "Metric deltas vs best conventional:"
    )

    report_lines.append(
        delta_df.to_string(
            index=False
        )
    )

    (
        REPORT_DIR
        / "sh_das_v6_report.txt"
    ).write_text(
        "\n".join(
            report_lines
        ),
        encoding="utf-8",
    )

    # ========================================================================
    # COMPLETION
    # ========================================================================

    print()
    print("=" * 75)
    print(
        "SH-DAS V6 COMPLETED"
    )
    print("=" * 75)

    print()
    print(
        "Reports saved to:"
    )

    print(
        REPORT_DIR
    )

    print()

    print(
        "Final SH-DAS V6 metrics:"
    )

    for metric, value in (
        v6_metrics.items()
    ):

        print(
            f"  {metric}: "
            f"{value:.6f}"
        )


# ============================================================================
# 15. ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()