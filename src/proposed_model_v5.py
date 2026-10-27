"""
SH-DAS V5
Performance-Optimized Drift-Aware Stacking

Purpose:
1. Build 5 conventional base models.
2. Introduce controlled feature drift.
3. Build meta-features from:
      - base-model probabilities
      - base-model predictions
      - feature-level drift indicators
      - model x drift interactions
4. Compare several meta-model candidates using ONLY adaptation data.
5. Tune the final classification threshold using ONLY adaptation data.
6. Freeze the selected V5 candidate.
7. Evaluate once on the untouched evaluation set.

Important:
The evaluation set is never used to select:
- the meta-model
- hyperparameters
- model weights
- threshold
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports" / "proposed_model_v5"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# CONFIGURATION
# ============================================================================

RANDOM_STATE = 42

TRAIN_SIZE = 0.60
ADAPT_SIZE = 0.20
EVAL_SIZE = 0.20

DRIFT_TRANSFORMS = {
    "tenure": 1.5,
    "MonthlyCharges": 1.2,
}

CATEGORICAL_STATUS = {
    "tenure": "High",
    "MonthlyCharges": "High",
}


# ============================================================================
# DATASET DISCOVERY
# ============================================================================

def find_dataset() -> Path:
    """
    Return the project's actual IBM Telco Customer Churn dataset.
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
# METRICS
# ============================================================================

def evaluate_model(
    y_true,
    y_prob,
    threshold=0.50,
):
    y_pred = (
        np.asarray(y_prob) >= threshold
    ).astype(int)

    return {
        "Accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "F1-score": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "ROC-AUC": roc_auc_score(
            y_true,
            y_prob,
        ),
    }


# ============================================================================
# THRESHOLD SEARCH
# ============================================================================

def find_best_threshold(
    y_true,
    probabilities,
):
    """
    Select threshold using adaptation data.

    Primary objective:
        maximize F1

    Tie-break:
        maximize ROC-AUC

    ROC-AUC is threshold-independent, so it is only used
    for reporting/tie handling between candidates.
    """

    best_threshold = 0.50
    best_f1 = -1.0

    thresholds = np.arange(
        0.20,
        0.71,
        0.01,
    )

    for threshold in thresholds:

        predictions = (
            np.asarray(probabilities)
            >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        if score > best_f1:

            best_f1 = score
            best_threshold = float(
                round(threshold, 2)
            )

    return best_threshold


# ============================================================================
# PREPROCESSING
# ============================================================================

def build_preprocessor(X):

    numeric_features = (
        X.select_dtypes(
            include=["int64", "float64"]
        )
        .columns
        .tolist()
    )

    categorical_features = (
        X.select_dtypes(
            include=["object", "category", "bool"]
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

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    return preprocessor


# ============================================================================
# BASE MODELS
# ============================================================================

def build_base_models():

    return {
        "Logistic Regression":
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                max_depth=6,
                min_samples_leaf=10,
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=250,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                min_samples_leaf=3,
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
# CONTROLLED DRIFT
# ============================================================================

def introduce_drift(X):

    X = X.copy()

    for feature, multiplier in DRIFT_TRANSFORMS.items():

        if feature in X.columns:

            X[feature] = (
                pd.to_numeric(
                    X[feature],
                    errors="coerce",
                )
                * multiplier
            )

            if feature == "tenure":
                X[feature] = X[feature].clip(
                    upper=72
                )

            if feature == "MonthlyCharges":
                X[feature] = X[feature].clip(
                    upper=120
                )

    return X


# ============================================================================
# FEATURE-LEVEL DRIFT
# ============================================================================

def compute_drift_features(
    reference,
    current,
):
    """
    Compute standardized and absolute mean-shift
    indicators for every numeric feature.

    These indicators are calculated against the
    training/reference distribution.
    """

    reference = reference.copy()
    current = current.copy()

    numeric_features = (
        reference.select_dtypes(
            include=["int64", "float64"]
        )
        .columns
        .tolist()
    )

    drift_abs = {}
    drift_z = {}

    for feature in numeric_features:

        ref_values = pd.to_numeric(
            reference[feature],
            errors="coerce",
        ).dropna()

        cur_values = pd.to_numeric(
            current[feature],
            errors="coerce",
        ).dropna()

        if len(ref_values) == 0:
            drift_abs[feature] = 0.0
            drift_z[feature] = 0.0
            continue

        ref_mean = ref_values.mean()
        ref_std = ref_values.std()

        cur_mean = (
            cur_values.mean()
            if len(cur_values) > 0
            else ref_mean
        )

        mean_shift = (
            cur_mean - ref_mean
        )

        drift_abs[feature] = float(
            abs(mean_shift)
        )

        if ref_std == 0 or pd.isna(ref_std):
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
# BASE-MODEL PREDICTIONS
# ============================================================================

def train_and_predict_base_models(
    X_train,
    y_train,
    X_target,
):

    models = {}
    probabilities = {}
    predictions = {}

    for name, base_model in build_base_models().items():

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
                    base_model,
                ),
            ]
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        prob = pipeline.predict_proba(
            X_target
        )[:, 1]

        pred = (
            prob >= 0.50
        ).astype(int)

        models[name] = pipeline
        probabilities[name] = prob
        predictions[name] = pred

    return (
        models,
        probabilities,
        predictions,
    )


# ============================================================================
# META-FEATURE CONSTRUCTION
# ============================================================================

def build_meta_features(
    probabilities,
    predictions,
    drift_abs,
    drift_z,
):
    """
    Build:
    - base probabilities
    - base predictions
    - drift summaries
    - model x drift interactions
    """

    result = pd.DataFrame(index=range(
        len(next(iter(probabilities.values())))
    ))

    # ------------------------------------------------------------------------
    # Base probabilities
    # ------------------------------------------------------------------------

    for name, values in probabilities.items():

        safe_name = (
            name.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        result[
            f"{safe_name}_prob"
        ] = values

    # ------------------------------------------------------------------------
    # Base predictions
    # ------------------------------------------------------------------------

    for name, values in predictions.items():

        safe_name = (
            name.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        result[
            f"{safe_name}_pred"
        ] = values

    # ------------------------------------------------------------------------
    # Drift features
    # ------------------------------------------------------------------------

    drift_feature_names = []

    for feature, value in drift_abs.items():

        col = (
            f"{feature}_drift_abs"
        )

        result[col] = value
        drift_feature_names.append(col)

    for feature, value in drift_z.items():

        col = (
            f"{feature}_drift_z"
        )

        result[col] = value

    # ------------------------------------------------------------------------
    # Global drift summaries
    # ------------------------------------------------------------------------

    abs_values = list(
        drift_abs.values()
    )

    z_values = list(
        drift_z.values()
    )

    result["global_drift_abs_mean"] = (
        float(np.mean(abs_values))
        if abs_values
        else 0.0
    )

    result["global_drift_abs_max"] = (
        float(np.max(abs_values))
        if abs_values
        else 0.0
    )

    result["global_drift_z_mean"] = (
        float(np.mean(
            np.abs(z_values)
        ))
        if z_values
        else 0.0
    )

    result["global_drift_z_max"] = (
        float(np.max(
            np.abs(z_values)
        ))
        if z_values
        else 0.0
    )

    # ------------------------------------------------------------------------
    # Model × drift interactions
    # ------------------------------------------------------------------------

    probability_columns = [
        c
        for c in result.columns
        if c.endswith("_prob")
    ]

    drift_columns = [
        c
        for c in result.columns
        if "drift_" in c
    ]

    for probability_column in probability_columns:

        model_prefix = probability_column.replace(
            "_prob",
            ""
        )

        for drift_column in drift_columns:

            result[
                f"{model_prefix}_x_{drift_column}"
            ] = (
                result[probability_column]
                * result[drift_column]
            )

    return result


# ============================================================================
# META-MODELS
# ============================================================================

def build_meta_candidates():

    candidates = {}

    # Logistic models across multiple regularization strengths
    for C in [
        0.03,
        0.10,
        0.30,
        1.0,
        3.0,
        10.0,
        30.0,
    ]:

        candidates[
            f"Meta Logistic C={C}"
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
                        C=C,
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

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

    return candidates


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():

    print()
    print("=" * 75)
    print(
        "SH-DAS V5 — PERFORMANCE-OPTIMIZED DRIFT-AWARE STACKING"
    )
    print("=" * 75)

    dataset_path = find_dataset()

    print()
    print(
        f"Dataset: {dataset_path}"
    )

    df = pd.read_csv(
        dataset_path
    )

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    df = df.dropna(
        axis=0,
        how="all",
    )

    if "customerID" in df.columns:
        df = df.drop(
            columns=["customerID"]
        )

    if "Churn" not in df.columns:
        raise ValueError(
            "Dataset must contain Churn."
        )

    y = (
        df["Churn"]
        .astype(str)
        .str.strip()
        .map({
            "Yes": 1,
            "No": 0,
        })
    )

    X = df.drop(
        columns=["Churn"]
    )

    # Convert TotalCharges when needed
    if "TotalCharges" in X.columns:
        X["TotalCharges"] = pd.to_numeric(
            X["TotalCharges"],
            errors="coerce",
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
        test_size=EVAL_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    adapt_fraction_of_remaining = (
        ADAPT_SIZE / (TRAIN_SIZE + ADAPT_SIZE)
    )

    (
        X_train,
        X_adapt,
        y_train,
        y_adapt,
    ) = train_test_split(
        X_temp,
        y_temp,
        test_size=adapt_fraction_of_remaining,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    print()
    print(
        f"Training records:    {len(X_train)}"
    )
    print(
        f"Adaptation records:  {len(X_adapt)}"
    )
    print(
        f"Evaluation records:  {len(X_eval)}"
    )

    # ========================================================================
    # CONTROLLED DRIFT
    # ========================================================================

    X_adapt_drift = introduce_drift(
        X_adapt
    )

    X_eval_drift = introduce_drift(
        X_eval
    )

    print()
    print(
        "Controlled drift introduced:"
    )
    print(
        "  tenure × 1.5, capped at 72"
    )
    print(
        "  MonthlyCharges × 1.2, capped at 120"
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
        base_models,
        adapt_probabilities,
        adapt_predictions,
    ) = train_and_predict_base_models(
        X_train,
        y_train,
        X_adapt_drift,
    )

    (
        _,
        eval_probabilities,
        eval_predictions,
    ) = train_and_predict_base_models(
        X_train,
        y_train,
        X_eval_drift,
    )

    # ========================================================================
    # DRIFT FEATURES
    # ========================================================================

    (
        adapt_drift_abs,
        adapt_drift_z,
    ) = compute_drift_features(
        X_train,
        X_adapt_drift,
    )

    (
        eval_drift_abs,
        eval_drift_z,
    ) = compute_drift_features(
        X_train,
        X_eval_drift,
    )

    # ========================================================================
    # META DATA
    # ========================================================================

    meta_adapt = build_meta_features(
        adapt_probabilities,
        adapt_predictions,
        adapt_drift_abs,
        adapt_drift_z,
    )

    meta_eval = build_meta_features(
        eval_probabilities,
        eval_predictions,
        eval_drift_abs,
        eval_drift_z,
    )

    print()
    print(
        f"Meta-feature count: {meta_adapt.shape[1]}"
    )

    # ========================================================================
    # INNER ADAPTATION SPLIT
    #
    # We use this to select the V5 meta-model and threshold
    # without touching the final evaluation set.
    # ========================================================================

    (
        meta_train,
        meta_valid,
        y_meta_train,
        y_meta_valid,
    ) = train_test_split(
        meta_adapt,
        y_adapt,
        test_size=0.30,
        stratify=y_adapt,
        random_state=RANDOM_STATE,
    )

    # ========================================================================
    # CANDIDATE SEARCH
    # ========================================================================

    print()
    print("=" * 75)
    print(
        "V5 META-MODEL SEARCH"
    )
    print("=" * 75)

    candidate_rows = []

    for candidate_name, candidate_model in (
        build_meta_candidates().items()
    ):

        print(
            f"Evaluating {candidate_name}..."
        )

        start = time.perf_counter()

        candidate_model.fit(
            meta_train,
            y_meta_train,
        )

        valid_prob = (
            candidate_model
            .predict_proba(
                meta_valid
            )[:, 1]
        )

        threshold = find_best_threshold(
            y_meta_valid,
            valid_prob,
        )

        metrics = evaluate_model(
            y_meta_valid,
            valid_prob,
            threshold,
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        candidate_rows.append(
            {
                "Candidate": candidate_name,
                "Threshold": threshold,
                "Accuracy": metrics["Accuracy"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1-score": metrics["F1-score"],
                "ROC-AUC": metrics["ROC-AUC"],
                "Time_sec": elapsed,
            }
        )

    candidates_df = pd.DataFrame(
        candidate_rows
    )

    candidates_df = candidates_df.sort_values(
        by=[
            "F1-score",
            "ROC-AUC",
            "Recall",
        ],
        ascending=False,
    )

    candidates_df.to_csv(
        REPORT_DIR
        / "v5_meta_candidate_search.csv",
        index=False,
    )

    print()
    print(
        candidates_df.to_string(
            index=False
        )
    )

    # ========================================================================
    # SELECT CANDIDATE
    # ========================================================================

    selected_row = (
        candidates_df.iloc[0]
    )

    selected_candidate = (
        selected_row["Candidate"]
    )

    selected_threshold = float(
        selected_row["Threshold"]
    )

    print()
    print(
        "=" * 75
    )
    print(
        "SELECTED V5 CANDIDATE"
    )
    print(
        "=" * 75
    )

    print(
        f"Candidate: {selected_candidate}"
    )

    print(
        f"Threshold: {selected_threshold:.2f}"
    )

    # ========================================================================
    # FIT SELECTED META-MODEL ON ALL ADAPTATION DATA
    # ========================================================================

    selected_model = build_meta_candidates()[
        selected_candidate
    ]

    selected_model.fit(
        meta_adapt,
        y_adapt,
    )

    # ========================================================================
    # FINAL EVALUATION — NEVER USED FOR SELECTION
    # ========================================================================

    final_probability = (
        selected_model
        .predict_proba(
            meta_eval
        )[:, 1]
    )

    v5_metrics = evaluate_model(
        y_eval,
        final_probability,
        selected_threshold,
    )

    # ========================================================================
    # BASELINE / CONVENTIONAL MODELS
    # ========================================================================

    comparison_rows = []

    for name, probabilities in (
        eval_probabilities.items()
    ):

        best_threshold = find_best_threshold(
            y_adapt,
            adapt_probabilities[name],
        )

        metrics = evaluate_model(
            y_eval,
            probabilities,
            best_threshold,
        )

        comparison_rows.append(
            {
                "Model": name,
                "Type": "Conventional",
                "Threshold": best_threshold,
                **metrics,
            }
        )

    comparison_rows.append(
        {
            "Model": "SH-DAS V5",
            "Type": "Proposed",
            "Threshold": selected_threshold,
            **v5_metrics,
        }
    )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_df.to_csv(
        REPORT_DIR
        / "six_model_v5_final_comparison.csv",
        index=False,
    )

    # ========================================================================
    # PRINT FINAL RESULTS
    # ========================================================================

    print()
    print("=" * 75)
    print(
        "SH-DAS V5 FINAL EVALUATION"
    )
    print("=" * 75)

    print(
        v5_metrics
    )

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
    # COMPARE V5 AGAINST BEST CONVENTIONAL BY EACH METRIC
    # ========================================================================

    conventional = comparison_df[
        comparison_df["Type"]
        == "Conventional"
    ].copy()

    deltas = {}

    for metric in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
    ]:

        best_value = conventional[
            metric
        ].max()

        v5_value = v5_metrics[
            metric
        ]

        deltas[
            f"{metric}_delta_vs_best_conventional"
        ] = float(
            v5_value - best_value
        )

    delta_df = pd.DataFrame(
        [
            {
                "Metric": k.replace(
                    "_delta_vs_best_conventional",
                    "",
                ),
                "V5_minus_best_conventional":
                    v,
            }
            for k, v in deltas.items()
        ]
    )

    delta_df.to_csv(
        REPORT_DIR
        / "sh_das_v5_vs_conventional.csv",
        index=False,
    )

    # ========================================================================
    # SAVE SUMMARY
    # ========================================================================

    summary = {
        "dataset_records": int(len(X)),
        "feature_count": int(X.shape[1]),
        "training_records": int(len(X_train)),
        "adaptation_records": int(len(X_adapt)),
        "evaluation_records": int(len(X_eval)),
        "selected_candidate": selected_candidate,
        "selected_threshold": selected_threshold,
        "v5_metrics": {
            k: float(v)
            for k, v in v5_metrics.items()
        },
        "metric_deltas_vs_best_conventional":
            deltas,
        "drift_transforms": DRIFT_TRANSFORMS,
        "selection_rule": [
            "Maximize adaptation F1",
            "Use ROC-AUC and Recall as tie-break information",
            "Do not use final evaluation records for candidate selection",
        ],
    }

    with (
        REPORT_DIR
        / "sh_das_v5_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 75)
    print(
        "SH-DAS V5 COMPLETED"
    )
    print("=" * 75)

    print()
    print(
        "Reports saved to:"
    )
    print(
        REPORT_DIR
    )


if __name__ == "__main__":
    main()