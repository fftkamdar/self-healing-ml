"""
SH-DAS V7
Row-Level Drift-Aware Probability Blending

Goal
----
Build a proposed model that balances:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - ROC-AUC

V7 improves on previous SH-DAS versions by:

1. Training heterogeneous base models.
2. Generating probability predictions.
3. Computing GLOBAL drift indicators.
4. Computing ROW-LEVEL drift indicators.
5. Searching probability-blend weights on adaptation data.
6. Testing several drift-aware mixture strategies.
7. Selecting the model using 5-fold CV on adaptation data only.
8. Optimizing the classification threshold on adaptation OOF predictions.
9. Freezing the selected V7 strategy.
10. Evaluating exactly once on the untouched evaluation set.

Final evaluation data is NEVER used for:
    - weight selection
    - strategy selection
    - threshold selection
    - model selection
"""

from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path

import joblib
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
    / "proposed_model_v7"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================================
# 2. CONFIGURATION
# ============================================================================

RANDOM_STATE = 42

EVALUATION_FRACTION = 0.20

DRIFT_TRANSFORMS = {
    "tenure": 1.5,
    "MonthlyCharges": 1.2,
}

THRESHOLD_MIN = 0.20
THRESHOLD_MAX = 0.70
THRESHOLD_STEP = 0.01


# ============================================================================
# 3. DATASET
# ============================================================================

def find_dataset() -> Path:

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
# 4. METRICS
# ============================================================================

def evaluate_model(
    y_true,
    probabilities,
    threshold: float,
) -> dict[str, float]:

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

    transformers = []

    if numeric_features:

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

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            )
        )

    if categorical_features:

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
# 7. BASE MODELS
# ============================================================================

def build_base_models():

    return {

        "Logistic Regression":
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                max_depth=6,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=250,
                min_samples_leaf=3,
                class_weight="balanced",
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
# 8. DRIFT INJECTION
# ============================================================================

def introduce_drift(
    X: pd.DataFrame,
) -> pd.DataFrame:

    X_drift = X.copy()

    if "tenure" in X_drift.columns:

        X_drift["tenure"] = (
            pd.to_numeric(
                X_drift["tenure"],
                errors="coerce",
            )
            * DRIFT_TRANSFORMS[
                "tenure"
            ]
        )

        X_drift["tenure"] = (
            X_drift["tenure"]
            .clip(upper=72)
        )

    if "MonthlyCharges" in X_drift.columns:

        X_drift["MonthlyCharges"] = (
            pd.to_numeric(
                X_drift["MonthlyCharges"],
                errors="coerce",
            )
            * DRIFT_TRANSFORMS[
                "MonthlyCharges"
            ]
        )

        X_drift["MonthlyCharges"] = (
            X_drift["MonthlyCharges"]
            .clip(upper=120)
        )

    return X_drift


# ============================================================================
# 9. GLOBAL DRIFT
# ============================================================================

def compute_global_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> dict:

    numeric_reference = (
        reference.select_dtypes(
            include=[
                "int64",
                "float64",
                "int32",
                "float32",
            ]
        )
    )

    numeric_current = (
        current.select_dtypes(
            include=[
                "int64",
                "float64",
                "int32",
                "float32",
            ]
        )
    )

    drift_rows = []

    for feature in numeric_reference.columns:

        if feature not in numeric_current.columns:
            continue

        ref_values = pd.to_numeric(
            numeric_reference[feature],
            errors="coerce",
        ).dropna()

        cur_values = pd.to_numeric(
            numeric_current[feature],
            errors="coerce",
        ).dropna()

        if ref_values.empty:
            continue

        reference_mean = float(
            ref_values.mean()
        )

        current_mean = float(
            cur_values.mean()
        )

        reference_std = float(
            ref_values.std()
        )

        mean_shift = (
            current_mean
            - reference_mean
        )

        if (
            reference_std == 0
            or np.isnan(reference_std)
        ):

            z_shift = 0.0

        else:

            z_shift = (
                mean_shift
                / reference_std
            )

        drift_rows.append(
            {
                "Feature": feature,
                "Reference_Mean":
                    reference_mean,
                "Current_Mean":
                    current_mean,
                "Absolute_Mean_Shift":
                    abs(mean_shift),
                "Drift_Z":
                    z_shift,
            }
        )

    drift_df = pd.DataFrame(
        drift_rows
    )

    if drift_df.empty:

        return {
            "rows": drift_df,
            "global_abs_mean": 0.0,
            "global_abs_max": 0.0,
            "global_z_mean": 0.0,
            "global_z_max": 0.0,
        }

    return {
        "rows": drift_df,
        "global_abs_mean":
            float(
                drift_df[
                    "Absolute_Mean_Shift"
                ].mean()
            ),
        "global_abs_max":
            float(
                drift_df[
                    "Absolute_Mean_Shift"
                ].max()
            ),
        "global_z_mean":
            float(
                np.abs(
                    drift_df[
                        "Drift_Z"
                    ]
                ).mean()
            ),
        "global_z_max":
            float(
                np.abs(
                    drift_df[
                        "Drift_Z"
                    ]
                ).max()
            ),
    }


# ============================================================================
# 10. ROW-LEVEL DRIFT
# ============================================================================

def compute_row_level_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> np.ndarray:
    """
    Calculate an individualized drift score for every row.

    For each numeric feature:

        |x - reference_mean| / reference_std

    The feature distances are normalized and averaged across
    numeric features.

    This gives V7 a customer-level drift signal instead of
    assigning the exact same drift value to every record.
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

    scores = np.zeros(
        len(current),
        dtype=float,
    )

    feature_count = 0

    for feature in reference_numeric.columns:

        if feature not in current_numeric.columns:
            continue

        reference_values = pd.to_numeric(
            reference_numeric[feature],
            errors="coerce",
        )

        current_values = pd.to_numeric(
            current_numeric[feature],
            errors="coerce",
        )

        mean = reference_values.mean()
        std = reference_values.std()

        if std == 0 or pd.isna(std):
            continue

        distance = (
            (
                current_values
                - mean
            )
            .abs()
            / std
        )

        distance = (
            distance
            .fillna(0.0)
            .to_numpy()
        )

        scores += distance
        feature_count += 1

    if feature_count > 0:

        scores = (
            scores
            / feature_count
        )

    # Compress large values into approximately 0–1.
    scores = (
        scores
        / (
            1.0
            + scores
        )
    )

    return scores


# ============================================================================
# 11. TRAIN BASE MODELS
# ============================================================================

def train_base_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_target: pd.DataFrame,
):

    probability_dict = {}

    prediction_dict = {}

    model_dict = {}

    for (
        model_name,
        estimator,
    ) in build_base_models().items():

        print(
            f"Training {model_name}..."
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

        probability_dict[
            model_name
        ] = probabilities

        prediction_dict[
            model_name
        ] = predictions

        model_dict[
            model_name
        ] = pipeline

    return (
        model_dict,
        probability_dict,
        prediction_dict,
    )


# ============================================================================
# 12. BLEND CANDIDATE
# ============================================================================

def make_blend(
    probabilities: dict[str, np.ndarray],
    weights: dict[str, float],
) -> np.ndarray:
    """
    Standard weighted probability blending.
    """

    result = np.zeros(
        len(
            next(
                iter(
                    probabilities.values()
                )
            )
        ),
        dtype=float,
    )

    total_weight = sum(
        weights.values()
    )

    if total_weight <= 0:

        raise ValueError(
            "Blend weights must have "
            "a positive total."
        )

    for model_name, weight in (
        weights.items()
    ):

        result += (
            weight
            * probabilities[
                model_name
            ]
        )

    result /= total_weight

    return result


# ============================================================================
# 13. DRIFT-AWARE BLEND
# ============================================================================

def make_drift_aware_blend(
    probabilities: dict[str, np.ndarray],
    base_weights: dict[str, float],
    focus_model: str,
    row_drift: np.ndarray,
    drift_strength: float,
) -> np.ndarray:
    """
    V7 mixture-of-experts idea.

    At low row-level drift:
        use the standard blend.

    At higher row-level drift:
        shift a proportion of probability weight toward
        the selected focus model.

    drift_strength:
        0.0 = no adjustment
        1.0 = strongest tested adjustment
    """

    base_probability = make_blend(
        probabilities,
        base_weights,
    )

    focus_probability = (
        probabilities[
            focus_model
        ]
    )

    adjustment = (
        np.asarray(row_drift)
        * float(drift_strength)
    )

    adjustment = np.clip(
        adjustment,
        0.0,
        1.0,
    )

    result = (
        (
            1.0
            - adjustment
        )
        * base_probability
        +
        adjustment
        * focus_probability
    )

    return np.clip(
        result,
        0.0,
        1.0,
    )


# ============================================================================
# 14. GENERATE BLEND WEIGHT CANDIDATES
# ============================================================================

def generate_weight_candidates():

    """
    Fast targeted blend search.

    Models:
        Logistic Regression
        SVM
        Gradient Boosting
    """

    return [
        # Mostly SVM
        (0.10, 0.80, 0.10),
        (0.20, 0.70, 0.10),
        (0.20, 0.60, 0.20),

        # Mostly Logistic Regression
        (0.70, 0.20, 0.10),
        (0.60, 0.30, 0.10),
        (0.50, 0.40, 0.10),

        # Mostly Gradient Boosting
        (0.10, 0.20, 0.70),
        (0.20, 0.20, 0.60),
        (0.20, 0.30, 0.50),

        # Balanced mixtures
        (0.40, 0.40, 0.20),
        (0.40, 0.30, 0.30),
        (0.30, 0.40, 0.30),
        (0.30, 0.30, 0.40),

        # Equal blend
        (1/3, 1/3, 1/3),

        # Strong SVM + Logistic
        (0.40, 0.50, 0.10),
        (0.50, 0.40, 0.10),

        # Strong Logistic + GB
        (0.50, 0.10, 0.40),
        (0.40, 0.10, 0.50),

        # Strong SVM + GB
        (0.10, 0.50, 0.40),
        (0.10, 0.40, 0.50),
    ]
# ============================================================================
# 15. CROSS-VALIDATED BLEND EVALUATION
# ============================================================================

def evaluate_blend_cv(
    probabilities: dict[str, np.ndarray],
    y: pd.Series,
    row_drift: np.ndarray,
    weights: dict[str, float],
    focus_model: str | None,
    drift_strength: float,
):
    """
    Evaluate one blending strategy using 5-fold CV.

    Base-model predictions are fixed because their training
    uses the separate model-training portion.

    Threshold is selected from the complete adaptation OOF
    predictions only after the folds are generated.
    """

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    number_of_rows = len(y)

    oof_probability = np.zeros(
        number_of_rows,
        dtype=float,
    )

    fold_rows = []

    for fold_number, (
        train_idx,
        valid_idx,
    ) in enumerate(
        cv.split(
            probabilities[
                "Logistic Regression"
            ],
            y,
        ),
        start=1,
    ):

        # The blend itself contains no trainable parameters
        # once its weights are fixed. We therefore evaluate
        # the fixed strategy on each held-out fold.

        fold_probabilities = {
            model_name:
                values[valid_idx]
            for (
                model_name,
                values
            ) in probabilities.items()
        }

        fold_drift = (
            row_drift[
                valid_idx
            ]
        )

        if focus_model is None:

            fold_probability = (
                make_blend(
                    fold_probabilities,
                    weights,
                )
            )

        else:

            fold_probability = (
                make_drift_aware_blend(
                    fold_probabilities,
                    weights,
                    focus_model,
                    fold_drift,
                    drift_strength,
                )
            )

        oof_probability[
            valid_idx
        ] = fold_probability

    threshold = (
        find_best_threshold(
            y,
            oof_probability,
        )
    )

    metrics = evaluate_model(
        y,
        oof_probability,
        threshold,
    )

    # Per-fold stability
    for fold_number, (
        train_idx,
        valid_idx,
    ) in enumerate(
        cv.split(
            probabilities[
                "Logistic Regression"
            ],
            y,
        ),
        start=1,
    ):

        fold_probabilities = {
            model_name:
                values[valid_idx]
            for (
                model_name,
                values
            ) in probabilities.items()
        }

        fold_drift = (
            row_drift[
                valid_idx
            ]
        )

        if focus_model is None:

            fold_probability = (
                make_blend(
                    fold_probabilities,
                    weights,
                )
            )

        else:

            fold_probability = (
                make_drift_aware_blend(
                    fold_probabilities,
                    weights,
                    focus_model,
                    fold_drift,
                    drift_strength,
                )
            )

        fold_metrics = (
            evaluate_model(
                y.iloc[valid_idx],
                fold_probability,
                threshold,
            )
        )

        fold_rows.append(
            {
                "Fold":
                    fold_number,

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

    fold_df = pd.DataFrame(
        fold_rows
    )

    return (
        metrics,
        threshold,
        fold_df,
    )


# ============================================================================
# 16. MAIN
# ============================================================================

def main():

    print()
    print("=" * 75)
    print(
        "SH-DAS V7 — ROW-LEVEL "
        "DRIFT-AWARE PROBABILITY BLENDING"
    )
    print("=" * 75)

    # ========================================================================
    # LOAD DATA
    # ========================================================================

    dataset_path = (
        find_dataset()
    )

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
            "The dataset must contain "
            "the Churn column."
        )

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
            "Unexpected target values "
            "found in Churn."
        )

    y = y.astype(int)

    X = df.drop(
        columns=[
            "Churn"
        ]
    ).copy()

    if "TotalCharges" in X.columns:

        X[
            "TotalCharges"
        ] = pd.to_numeric(
            X[
                "TotalCharges"
            ],
            errors="coerce",
        )

    print(
        f"Dataset shape: {X.shape}"
    )

    # ========================================================================
    # TRAIN / ADAPTATION / EVALUATION
    # ========================================================================

    (
        X_development,
        X_eval,
        y_development,
        y_eval,
    ) = train_test_split(
        X,
        y,
        test_size=EVALUATION_FRACTION,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    (
        X_train,
        X_adaptation,
        y_train,
        y_adaptation,
    ) = train_test_split(
        X_development,
        y_development,
        test_size=0.25,
        stratify=y_development,
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
    # DRIFT
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
    # GLOBAL DRIFT DIAGNOSIS
    # ========================================================================

    global_drift = (
        compute_global_drift(
            X_train,
            X_adaptation_drift,
        )
    )

    drift_table = (
        global_drift["rows"]
        .sort_values(
            by="Absolute_Mean_Shift",
            ascending=False,
        )
    )

    drift_table.to_csv(
        REPORT_DIR
        / "v7_global_drift.csv",
        index=False,
    )

    print()
    print(
        "Global drift summary:"
    )

    print(
        drift_table.head(10).to_string(
            index=False
        )
    )

    # ========================================================================
    # ROW-LEVEL DRIFT
    # ========================================================================

    adaptation_row_drift = (
        compute_row_level_drift(
            X_train,
            X_adaptation_drift,
        )
    )

    evaluation_row_drift = (
        compute_row_level_drift(
            X_train,
            X_eval_drift,
        )
    )

    pd.DataFrame(
        {
            "row_drift_score":
                adaptation_row_drift
        }
    ).to_csv(
        REPORT_DIR
        / "v7_adaptation_row_drift.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "row_drift_score":
                evaluation_row_drift
        }
    ).to_csv(
        REPORT_DIR
        / "v7_evaluation_row_drift.csv",
        index=False,
    )

    print()
    print(
        "Row-level drift:"
    )

    print(
        f"  Adaptation mean: "
        f"{adaptation_row_drift.mean():.4f}"
    )

    print(
        f"  Adaptation max:  "
        f"{adaptation_row_drift.max():.4f}"
    )

    print(
        f"  Evaluation mean: "
        f"{evaluation_row_drift.mean():.4f}"
    )

    print(
        f"  Evaluation max:  "
        f"{evaluation_row_drift.max():.4f}"
    )

    # ========================================================================
    # BASE MODELS
    # ========================================================================

    print()
    print("=" * 75)
    print(
        "TRAINING BASE MODELS"
    )
    print("=" * 75)

    (
        adaptation_models,
        adaptation_probabilities,
        adaptation_predictions,
    ) = train_base_models(
        X_train,
        y_train,
        X_adaptation_drift,
    )

    (
        evaluation_models,
        evaluation_probabilities,
        evaluation_predictions,
    ) = train_base_models(
        X_train,
        y_train,
        X_eval_drift,
    )

    # ========================================================================
    # SAVE BASE MODEL EVALUATIONS
    # ========================================================================

    base_rows = []

    for model_name, probabilities in (
        evaluation_probabilities.items()
    ):

        threshold = (
            find_best_threshold(
                y_adaptation,
                adaptation_probabilities[
                    model_name
                ],
            )
        )

        metrics = (
            evaluate_model(
                y_eval,
                probabilities,
                threshold,
            )
        )

        base_rows.append(
            {
                "Model":
                    model_name,
                "Type":
                    "Conventional",
                "Threshold":
                    threshold,
                **metrics,
            }
        )

    base_results_df = pd.DataFrame(
        base_rows
    )

    base_results_df.to_csv(
        REPORT_DIR
        / "v7_base_model_results.csv",
        index=False,
    )

    # ========================================================================
    # WEIGHT SEARCH
    # ========================================================================

    print()
    print("=" * 75)
    print(
        "V7 CROSS-VALIDATED BLEND SEARCH"
    )
    print("=" * 75)

    blend_candidates = (
        generate_weight_candidates()
    )

    print(
        f"Weight candidates: "
        f"{len(blend_candidates)}"
    )

    search_rows = []

    strongest_base_models = [
        "Logistic Regression",
        "SVM",
        "Gradient Boosting",
    ]

    # ------------------------------------------------------------------------
    # Standard blends
    # ------------------------------------------------------------------------

    candidate_counter = 0

    for (
        lr_weight,
        svm_weight,
        gb_weight,
    ) in blend_candidates:

        weights = {
            "Logistic Regression":
                float(lr_weight),

            "SVM":
                float(svm_weight),

            "Gradient Boosting":
                float(gb_weight),
        }

        candidate_counter += 1

        if candidate_counter % 100 == 0:

            print(
                f"Processed "
                f"{candidate_counter}/"
                f"{len(blend_candidates)} "
                "standard blends..."
            )

        (
            metrics,
            threshold,
            fold_df,
        ) = evaluate_blend_cv(
            adaptation_probabilities,
            y_adaptation,
            adaptation_row_drift,
            weights,
            None,
            0.0,
        )

        search_rows.append(
            {
                "Strategy":
                    "Standard Blend",

                "LR_weight":
                    lr_weight,

                "SVM_weight":
                    svm_weight,

                "GB_weight":
                    gb_weight,

                "Focus_Model":
                    "None",

                "Drift_Strength":
                    0.0,

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

                "CV_F1_STD":
                    fold_df[
                        "F1-score"
                    ].std(),

                "CV_AUC_MEAN":
                    fold_df[
                        "ROC-AUC"
                    ].mean(),
            }
        )

    # ------------------------------------------------------------------------
    # Drift-aware blends
    # ------------------------------------------------------------------------

    print()
    print(
        "Searching row-level drift-aware blends..."
    )

    drift_strength_values = [
    0.20,
    0.40,
    0.60,
]

    for (
        lr_weight,
        svm_weight,
        gb_weight,
    ) in blend_candidates:

        weights = {
            "Logistic Regression":
                float(lr_weight),

            "SVM":
                float(svm_weight),

            "Gradient Boosting":
                float(gb_weight),
        }

        for focus_model in (
            strongest_base_models
        ):

            for drift_strength in (
                drift_strength_values
            ):

                (
                    metrics,
                    threshold,
                    fold_df,
                ) = evaluate_blend_cv(
                    adaptation_probabilities,
                    y_adaptation,
                    adaptation_row_drift,
                    weights,
                    focus_model,
                    drift_strength,
                )

                search_rows.append(
                    {
                        "Strategy":
                            "Drift-Aware Blend",

                        "LR_weight":
                            lr_weight,

                        "SVM_weight":
                            svm_weight,

                        "GB_weight":
                            gb_weight,

                        "Focus_Model":
                            focus_model,

                        "Drift_Strength":
                            drift_strength,

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

                        "CV_F1_STD":
                            fold_df[
                                "F1-score"
                            ].std(),

                        "CV_AUC_MEAN":
                            fold_df[
                                "ROC-AUC"
                            ].mean(),
                    }
                )

    search_df = pd.DataFrame(
        search_rows
    )

    # ========================================================================
    # SEARCH RANKING
    # ========================================================================

    search_df = (
        search_df
        .sort_values(
            by=[
                "F1-score",
                "ROC-AUC",
                "Recall",
                "Accuracy",
            ],
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    search_df.to_csv(
        REPORT_DIR
        / "v7_blend_search.csv",
        index=False,
    )

    print()
    print(
        "Top adaptation candidates:"
    )

    print(
        search_df.head(15).to_string(
            index=False
        )
    )

    # ========================================================================
    # SELECT V7 STRATEGY
    # ========================================================================

    selected = (
        search_df.iloc[0]
    )

    selected_weights = {
        "Logistic Regression":
            float(
                selected[
                    "LR_weight"
                ]
            ),

        "SVM":
            float(
                selected[
                    "SVM_weight"
                ]
            ),

        "Gradient Boosting":
            float(
                selected[
                    "GB_weight"
                ]
            ),
    }

    selected_focus_model = (
        None
        if str(
            selected[
                "Focus_Model"
            ]
        ) == "None"
        else str(
            selected[
                "Focus_Model"
            ]
        )
    )

    selected_drift_strength = float(
        selected[
            "Drift_Strength"
        ]
    )

    selected_threshold = float(
        selected[
            "Threshold"
        ]
    )

    # ========================================================================
    # PRINT SELECTED STRATEGY
    # ========================================================================

    print()
    print("=" * 75)
    print(
        "SELECTED V7 STRATEGY"
    )
    print("=" * 75)

    print(
        f"LR weight: "
        f"{selected_weights['Logistic Regression']:.3f}"
    )

    print(
        f"SVM weight: "
        f"{selected_weights['SVM']:.3f}"
    )

    print(
        f"GB weight: "
        f"{selected_weights['Gradient Boosting']:.3f}"
    )

    print(
        f"Focus model: "
        f"{selected_focus_model}"
    )

    print(
        f"Drift strength: "
        f"{selected_drift_strength:.2f}"
    )

    print(
        f"Threshold: "
        f"{selected_threshold:.2f}"
    )

    print(
        f"Adaptation F1: "
        f"{selected['F1-score']:.4f}"
    )

    print(
        f"Adaptation ROC-AUC: "
        f"{selected['ROC-AUC']:.4f}"
    )

    # ========================================================================
    # FINAL V7 EVALUATION
    # ========================================================================

    print()
    print("=" * 75)
    print(
        "SH-DAS V7 FINAL EVALUATION"
    )
    print("=" * 75)

    if selected_focus_model is None:

        final_probability = (
            make_blend(
                evaluation_probabilities,
                selected_weights,
            )
        )

    else:

        final_probability = (
            make_drift_aware_blend(
                evaluation_probabilities,
                selected_weights,
                selected_focus_model,
                evaluation_row_drift,
                selected_drift_strength,
            )
        )

    v7_metrics = (
        evaluate_model(
            y_eval,
            final_probability,
            selected_threshold,
        )
    )

    print(
        v7_metrics
    )

    # ========================================================================
    # FINAL COMPARISON
    # ========================================================================

    comparison_rows = (
        base_rows.copy()
    )

    comparison_rows.append(
        {
            "Model":
                "SH-DAS V7",

            "Type":
                "Proposed",

            "Threshold":
                selected_threshold,

            **v7_metrics,
        }
    )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_df.to_csv(
        REPORT_DIR
        / "six_model_v7_final_comparison.csv",
        index=False,
    )

    # ========================================================================
    # DELTAS
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

        v7_value = float(
            v7_metrics[
                metric
            ]
        )

        delta_rows.append(
            {
                "Metric":
                    metric,

                "Best_Conventional":
                    best_conventional,

                "SH_DAS_V7":
                    v7_value,

                "V7_minus_Best":
                    v7_value
                    - best_conventional,
            }
        )

    delta_df = pd.DataFrame(
        delta_rows
    )

    delta_df.to_csv(
        REPORT_DIR
        / "sh_das_v7_vs_conventional.csv",
        index=False,
    )

    # ========================================================================
    # SAVE SELECTED MODEL INFORMATION
    # ========================================================================

    selected_strategy = {
        "model_name":
            "SH-DAS V7",

        "base_models": [
            "Logistic Regression",
            "SVM",
            "Gradient Boosting",
        ],

        "weights":
            selected_weights,

        "focus_model":
            selected_focus_model,

        "drift_strength":
            selected_drift_strength,

        "threshold":
            selected_threshold,

        "adaptation_metrics": {
            metric:
                float(
                    selected[
                        metric
                    ]
                )
            for metric in [
                "Accuracy",
                "Precision",
                "Recall",
                "F1-score",
                "ROC-AUC",
            ]
        },

        "final_metrics":
            v7_metrics,

        "selection_rule": [
            "5-fold adaptation cross-validation",
            "Primary objective: adaptation F1-score",
            "Tie-break: adaptation ROC-AUC",
            "Final tie-break: adaptation Recall",
            "Threshold selected from adaptation OOF predictions",
            "Final evaluation set never used for selection",
        ],
    }

    with (
        REPORT_DIR
        / "sh_das_v7_strategy.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            selected_strategy,
            file,
            indent=2,
        )

    # ========================================================================
    # SAVE PROBABILITIES
    # ========================================================================

    probability_df = pd.DataFrame(
        {
            "y_true":
                y_eval.values,

            "SH_DAS_V7_probability":
                final_probability,

            "row_level_drift":
                evaluation_row_drift,
        }
    )

    probability_df.to_csv(
        REPORT_DIR
        / "sh_das_v7_evaluation_predictions.csv",
        index=False,
    )

    # ========================================================================
    # SAVE MODEL
    # ========================================================================

    joblib.dump(
        {
            "strategy":
                selected_strategy,

            "base_models":
                {
                    name:
                        evaluation_models[
                            name
                        ]

                    for name
                    in selected_weights
                },
        },
        REPORT_DIR
        / "sh_das_v7_model_bundle.joblib",
    )

    # ========================================================================
    # FINAL REPORT
    # ========================================================================

    report_lines = []

    report_lines.append(
        "SH-DAS V7 — FINAL REPORT"
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
        f"Feature count: {X.shape[1]}"
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
        "Selected V7 strategy:"
    )

    report_lines.append(
        str(
            selected_strategy
        )
    )

    report_lines.append(
        ""
    )

    report_lines.append(
        "Final metrics:"
    )

    for metric, value in (
        v7_metrics.items()
    ):

        report_lines.append(
            f"  {metric}: "
            f"{value:.6f}"
        )

    report_lines.append(
        ""
    )

    report_lines.append(
        "Final comparison:"
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
        "Deltas vs best conventional:"
    )

    report_lines.append(
        delta_df.to_string(
            index=False
        )
    )

    (
        REPORT_DIR
        / "sh_das_v7_report.txt"
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
        "SH-DAS V7 COMPLETED"
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
        "Final SH-DAS V7 metrics:"
    )

    for metric, value in (
        v7_metrics.items()
    ):

        print(
            f"  {metric}: "
            f"{value:.6f}"
        )


# ============================================================================
# 17. ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()