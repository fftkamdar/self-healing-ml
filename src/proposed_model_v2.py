"""
SH-DAE V2
---------
Self-Healing Drift-Aware Ensemble

Version 2 improvements over V1:

1. Uses adaptation F1, Recall and ROC-AUC rather than ROC-AUC alone.
2. Uses a milder drift-exposure penalty.
3. Builds a weighted probability ensemble.
4. Learns an operating threshold ONLY on the adaptation set.
5. Evaluates the final model on a completely separate held-out set.

This is an academic research prototype.
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
    / "proposed_model_v2"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_STATE = 42

TRAIN_SIZE = 0.60
ADAPTATION_SIZE = 0.20
EVALUATION_SIZE = 0.20

N_PERMUTATION_REPEATS = 7

# V1 used 2.0.
# V2 deliberately uses a milder penalty.
DRIFT_PENALTY = 0.50

EPSILON = 1e-6

THRESHOLD_VALUES = np.arange(
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

    df = pd.read_csv(DATA_PATH)

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

    if y.isna().any():
        raise ValueError(
            "Unexpected values found in Churn."
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
# FIVE MODELS
# =============================================================================

def build_models():

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

    reference = pd.to_numeric(
        reference,
        errors="coerce",
    ).dropna()

    current = pd.to_numeric(
        current,
        errors="coerce",
    ).dropna()

    if reference.empty or current.empty:
        return 0.0

    boundaries = np.unique(
        reference.quantile(
            np.linspace(
                0,
                1,
                11,
            )
        ).to_numpy()
    )

    if len(boundaries) < 2:
        return 0.0

    boundaries[0] = -np.inf
    boundaries[-1] = np.inf

    ref_bins = pd.cut(
        reference,
        bins=boundaries,
        include_lowest=True,
    )

    cur_bins = pd.cut(
        current,
        bins=boundaries,
        include_lowest=True,
    )

    ref_prop = (
        ref_bins
        .value_counts(
            normalize=True,
            sort=False,
        )
        .to_numpy()
    )

    cur_prop = (
        cur_bins
        .value_counts(
            normalize=True,
            sort=False,
        )
        .to_numpy()
    )

    ref_prop = np.clip(
        ref_prop,
        EPSILON,
        None,
    )

    cur_prop = np.clip(
        cur_prop,
        EPSILON,
        None,
    )

    return float(
        np.sum(
            (
                cur_prop - ref_prop
            )
            * np.log(
                cur_prop / ref_prop
            )
        )
    )


def calculate_drift_table(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for feature in reference.columns:

        if pd.api.types.is_numeric_dtype(
            reference[feature]
        ):

            psi = calculate_numeric_psi(
                reference[feature],
                current[feature],
            )

        else:

            ref_counts = (
                reference[feature]
                .astype(str)
                .value_counts(
                    normalize=True
                )
            )

            cur_counts = (
                current[feature]
                .astype(str)
                .value_counts(
                    normalize=True
                )
            )

            categories = sorted(
                set(ref_counts.index)
                | set(cur_counts.index)
            )

            ref_values = np.array(
                [
                    ref_counts.get(
                        category,
                        0.0,
                    )
                    for category in categories
                ]
            )

            cur_values = np.array(
                [
                    cur_counts.get(
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
# TRAIN MODEL
# =============================================================================

def train_pipeline(
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
# FEATURE IMPORTANCE
# =============================================================================

def get_feature_importance(
    pipeline,
    X_adaptation,
    y_adaptation,
):

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
            "Importance": (
                result.importances_mean
            ),
            "Importance_Std": (
                result.importances_std
            ),
        }
    )


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold=0.50,
):

    predictions = (
        probabilities >= threshold
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
# MAIN
# =============================================================================

def main():

    print("=" * 75)
    print(
        "SELF-HEALING ML — SH-DAE V2"
    )
    print("=" * 75)

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------

    X, y = load_dataset()

    print(
        f"\nDataset: {len(X)} rows, "
        f"{X.shape[1]} features"
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

    eval_fraction = (
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
            test_size=eval_fraction,
            stratify=y_remaining,
            random_state=RANDOM_STATE,
        )
    )

    print(
        f"Training records:   {len(X_train)}"
    )

    print(
        f"Adaptation records: {len(X_adaptation)}"
    )

    print(
        f"Evaluation records: {len(X_eval)}"
    )

    # -------------------------------------------------------------------------
    # Drift
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

    drift_table = calculate_drift_table(
        X_train,
        X_adaptation_drifted,
    )

    drift_table.to_csv(
        REPORT_DIR
        / "proposed_model_v2_drift.csv",
        index=False,
    )

    print(
        "\nTop drifted features:"
    )

    print(
        drift_table
        .sort_values(
            "PSI",
            ascending=False,
        )
        .head(10)
        .to_string(index=False)
    )

    # -------------------------------------------------------------------------
    # Train five models
    # -------------------------------------------------------------------------

    models = build_models()

    pipelines = {}

    adaptation_probabilities = {}

    evaluation_probabilities = {}

    model_info = {}

    print(
        "\n" + "=" * 75
    )

    print(
        "TRAINING BASE MODELS"
    )

    print(
        "=" * 75
    )

    for model_name, model in models.items():

        print(
            f"\nTraining {model_name}..."
        )

        pipeline = train_pipeline(
            model,
            X_train,
            y_train,
        )

        pipelines[
            model_name
        ] = pipeline

        adapt_prob = (
            pipeline
            .predict_proba(
                X_adaptation_drifted
            )[:, 1]
        )

        eval_prob = (
            pipeline
            .predict_proba(
                X_eval_drifted
            )[:, 1]
        )

        adaptation_probabilities[
            model_name
        ] = adapt_prob

        evaluation_probabilities[
            model_name
        ] = eval_prob

        adapt_metrics = calculate_metrics(
            y_adaptation,
            adapt_prob,
            threshold=0.50,
        )

        importance = get_feature_importance(
            pipeline,
            X_adaptation_drifted,
            y_adaptation,
        )

        model_info[
            model_name
        ] = {
            "adapt_metrics": adapt_metrics,
            "importance": importance,
        }

        print(
            f"Adaptation Accuracy: "
            f"{adapt_metrics['Accuracy']:.6f}"
        )

        print(
            f"Adaptation Precision: "
            f"{adapt_metrics['Precision']:.6f}"
        )

        print(
            f"Adaptation Recall: "
            f"{adapt_metrics['Recall']:.6f}"
        )

        print(
            f"Adaptation F1: "
            f"{adapt_metrics['F1-score']:.6f}"
        )

        print(
            f"Adaptation ROC-AUC: "
            f"{adapt_metrics['ROC-AUC']:.6f}"
        )

    # -------------------------------------------------------------------------
    # Drift exposure
    # -------------------------------------------------------------------------

    drift_lookup = dict(
        zip(
            drift_table["Feature"],
            drift_table["PSI"],
        )
    )

    weight_rows = []

    for model_name, info in (
        model_info.items()
    ):

        importance_df = (
            info["importance"]
            .copy()
        )

        importance_df["PSI"] = (
            importance_df["Feature"]
            .map(
                drift_lookup
            )
            .fillna(0.0)
        )

        importance_df[
            "Absolute_Importance"
        ] = importance_df[
            "Importance"
        ].abs()

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

        metrics = info[
            "adapt_metrics"
        ]

        # Composite adaptation score.
        #
        # F1 receives the largest contribution because it balances
        # precision and recall for this imbalanced classification task.
        performance_score = (
            0.50 * metrics["F1-score"]
            + 0.30 * metrics["ROC-AUC"]
            + 0.20 * metrics["Recall"]
        )

        # Mild drift penalty.
        stability_factor = (
            1.0
            / (
                1.0
                + DRIFT_PENALTY
                * drift_exposure
            )
        )

        stability_score = (
            performance_score
            * stability_factor
        )

        weight_rows.append(
            {
                "Model": model_name,
                "Adaptation_F1": metrics[
                    "F1-score"
                ],
                "Adaptation_Recall": metrics[
                    "Recall"
                ],
                "Adaptation_ROC_AUC": metrics[
                    "ROC-AUC"
                ],
                "Drift_Exposure": drift_exposure,
                "Performance_Score": performance_score,
                "Stability_Factor": stability_factor,
                "Stability_Score": stability_score,
            }
        )

    weights_df = pd.DataFrame(
        weight_rows
    )

    total_score = (
        weights_df[
            "Stability_Score"
        ].sum()
    )

    weights_df[
        "Weight"
    ] = (
        weights_df[
            "Stability_Score"
        ]
        / total_score
    )

    weights_df = weights_df.sort_values(
        "Weight",
        ascending=False,
    ).reset_index(drop=True)

    weights_df.to_csv(
        REPORT_DIR
        / "sh_dae_v2_model_weights.csv",
        index=False,
    )

    print(
        "\n" + "=" * 75
    )

    print(
        "V2 ADAPTIVE MODEL WEIGHTS"
    )

    print(
        "=" * 75
    )

    print(
        weights_df.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Create weighted ensemble
    # -------------------------------------------------------------------------

    weight_lookup = dict(
        zip(
            weights_df["Model"],
            weights_df["Weight"],
        )
    )

    adaptation_ensemble = np.zeros(
        len(X_adaptation_drifted)
    )

    evaluation_ensemble = np.zeros(
        len(X_eval_drifted)
    )

    for model_name in models:

        weight = weight_lookup[
            model_name
        ]

        adaptation_ensemble += (
            weight
            * adaptation_probabilities[
                model_name
            ]
        )

        evaluation_ensemble += (
            weight
            * evaluation_probabilities[
                model_name
            ]
        )

    # -------------------------------------------------------------------------
    # Learn ensemble threshold ONLY from adaptation data
    # -------------------------------------------------------------------------

    threshold_rows = []

    for threshold in THRESHOLD_VALUES:

        metrics = calculate_metrics(
            y_adaptation,
            adaptation_ensemble,
            threshold=float(threshold),
        )

        threshold_rows.append(
            {
                "Threshold": float(
                    threshold
                ),
                **metrics,
            }
        )

    threshold_df = pd.DataFrame(
        threshold_rows
    )

    threshold_df = threshold_df.sort_values(
        by=[
            "F1-score",
            "Recall",
            "Precision",
        ],
        ascending=False,
    ).reset_index(
        drop=True
    )

    best_threshold = float(
        threshold_df.iloc[0][
            "Threshold"
        ]
    )

    threshold_df.to_csv(
        REPORT_DIR
        / "sh_dae_v2_threshold_search.csv",
        index=False,
    )

    print(
        "\n" + "=" * 75
    )

    print(
        "ADAPTIVE THRESHOLD"
    )

    print(
        "=" * 75
    )

    print(
        f"Selected threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Adaptation F1 at selected threshold: "
        f"{threshold_df.iloc[0]['F1-score']:.6f}"
    )

    print(
        f"Adaptation Recall at selected threshold: "
        f"{threshold_df.iloc[0]['Recall']:.6f}"
    )

    # -------------------------------------------------------------------------
    # Final six-model evaluation
    # -------------------------------------------------------------------------

    results = []

    for model_name in models:

        metrics = calculate_metrics(
            y_eval,
            evaluation_probabilities[
                model_name
            ],
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

    sh_dae_metrics = calculate_metrics(
        y_eval,
        evaluation_ensemble,
        threshold=best_threshold,
    )

    results.append(
        {
            "Model": "SH-DAE V2 (Proposed)",
            **sh_dae_metrics,
            "Threshold": best_threshold,
            "Drift_Aware": True,
        }
    )

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        REPORT_DIR
        / "six_model_v2_final_comparison.csv",
        index=False,
    )

    print(
        "\n" + "=" * 75
    )

    print(
        "FINAL SIX-MODEL COMPARISON — V2"
    )

    print(
        "=" * 75
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Compare proposed model against best conventional value per metric.
    # -------------------------------------------------------------------------

    conventional = results_df[
        results_df["Model"]
        != "SH-DAE V2 (Proposed)"
    ]

    proposed = results_df[
        results_df["Model"]
        == "SH-DAE V2 (Proposed)"
    ].iloc[0]

    comparison_rows = []

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

        proposed_value = proposed[
            metric
        ]

        comparison_rows.append(
            {
                "Metric": metric,
                "Best_Conventional_Value": (
                    best_value
                ),
                "SH_DAE_V2_Value": (
                    proposed_value
                ),
                "Delta": (
                    proposed_value
                    - best_value
                ),
            }
        )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_df.to_csv(
        REPORT_DIR
        / "sh_dae_v2_vs_conventional.csv",
        index=False,
    )

    print(
        "\n" + "=" * 75
    )

    print(
        "V2 DELTA AGAINST BEST CONVENTIONAL VALUE"
    )

    print(
        "=" * 75
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
        figsize=(11, 6)
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
        "Five Conventional Models vs SH-DAE V2"
    )

    plt.xticks(
        rotation=30,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR
        / "six_model_v2_roc_auc.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # -------------------------------------------------------------------------
    # Save summary
    # -------------------------------------------------------------------------

    summary = {
        "model_name": "SH-DAE V2",
        "full_name": (
            "Self-Healing Drift-Aware Ensemble"
        ),
        "base_models": list(
            models.keys()
        ),
        "drifted_features": [
            "tenure",
            "MonthlyCharges",
        ],
        "drift_penalty": DRIFT_PENALTY,
        "permutation_repeats": (
            N_PERMUTATION_REPEATS
        ),
        "selected_threshold": (
            best_threshold
        ),
        "weighting_strategy": (
            "Adaptation F1 + ROC-AUC + Recall "
            "with mild drift-exposure penalty"
        ),
        "evaluation_protocol": (
            "Weights and threshold selected using "
            "training/adaptation data; final metrics "
            "measured on separate held-out drifted "
            "evaluation data."
        ),
    }

    with (
        REPORT_DIR
        / "sh_dae_v2_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print(
        "\n" + "=" * 75
    )

    print(
        "SH-DAE V2 EXPERIMENT COMPLETED"
    )

    print(
        "=" * 75
    )

    print(
        f"\nReports saved to:\n{REPORT_DIR}"
    )


if __name__ == "__main__":
    main()