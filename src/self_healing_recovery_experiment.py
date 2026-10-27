"""
SELF-HEALING RECOVERY EXPERIMENT
--------------------------------

Purpose:
    Demonstrate the complete self-healing workflow:

        Normal Model
             ↓
        Drift occurs
             ↓
        Drift detected
             ↓
        Recovery triggered
             ↓
        Model retrained/adapted
             ↓
        Recovered model evaluated

The experiment compares:

    1. Static conventional Logistic Regression
       before recovery

    2. Static conventional Logistic Regression
       after recovery

    3. SH-DAS proposed model before recovery

    4. SH-DAS proposed model after recovery

The same final evaluation records are used before and after recovery.

Controlled drift:
    tenure × 1.5, capped at 72
    MonthlyCharges × 1.2, capped at 120

This is an academic prototype using simulated drift.
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
    / "self_healing_recovery"
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

PSI_ALERT_THRESHOLD = 0.25

DRIFT_FEATURES = [
    "tenure",
    "MonthlyCharges",
]

THRESHOLDS = np.arange(
    0.20,
    0.81,
    0.01,
)

EPSILON = 1e-6


# =============================================================================
# LOAD DATA
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
            "Unexpected values were found in Churn."
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
# FIVE CONVENTIONAL MODELS
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
# PSI
# =============================================================================

def calculate_numeric_psi(
    reference,
    current,
):

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
                cur_prop
                - ref_prop
            )
            * np.log(
                cur_prop
                / ref_prop
            )
        )
    )


def calculate_psi_table(
    reference,
    current,
):

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
# TRAINING
# =============================================================================

def train_model(
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
# BUILD OOF META DATA
# =============================================================================

def build_oof_meta_dataset(
    X_train,
    y_train,
):

    print(
        "\n"
        + "=" * 75
    )

    print(
        "BUILDING DRIFT-AWARE OOF META-DATASET"
    )

    print(
        "=" * 75
    )

    models = build_models()

    splitter = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    parts = []
    targets = []

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

        drift_reference = {
            feature: {
                "mean": float(
                    pd.to_numeric(
                        X_fold_train[
                            feature
                        ],
                        errors="coerce",
                    ).mean()
                ),
                "std": float(
                    max(
                        pd.to_numeric(
                            X_fold_train[
                                feature
                            ],
                            errors="coerce",
                        ).std(),
                        EPSILON,
                    )
                ),
            }
            for feature in DRIFT_FEATURES
            if feature in X_fold_train.columns
        }

        fold_meta = pd.DataFrame(
            index=X_fold_valid_drifted.index
        )

        # Drift features.
        for feature in DRIFT_FEATURES:

            if feature not in X_fold_valid_drifted.columns:
                continue

            if feature not in drift_reference:
                continue

            values = pd.to_numeric(
                X_fold_valid_drifted[
                    feature
                ],
                errors="coerce",
            )

            mean = drift_reference[
                feature
            ]["mean"]

            std = drift_reference[
                feature
            ]["std"]

            z = (
                values - mean
            ) / std

            z = z.fillna(0.0)

            fold_meta[
                f"{feature}_drift_z"
            ] = z

            fold_meta[
                f"{feature}_drift_abs"
            ] = z.abs()

        # Five model predictions.
        fold_pipelines = {}

        for model_name, model in (
            models.items()
        ):

            fold_pipelines[
                model_name
            ] = train_model(
                model,
                X_fold_train,
                y_fold_train,
            )

            probability = (
                fold_pipelines[
                    model_name
                ]
                .predict_proba(
                    X_fold_valid_drifted
                )[:, 1]
            )

            safe_name = (
                model_name
                .replace(
                    " ",
                    "_",
                )
                .replace(
                    "-",
                    "_",
                )
            )

            fold_meta[
                f"{safe_name}_prob"
            ] = probability

            for drift_feature in DRIFT_FEATURES:

                drift_column = (
                    f"{drift_feature}_drift_abs"
                )

                if drift_column in fold_meta.columns:

                    fold_meta[
                        f"{safe_name}_x_"
                        f"{drift_feature}_drift_abs"
                    ] = (
                        probability
                        * fold_meta[
                            drift_column
                        ]
                    )

        parts.append(
            fold_meta.reset_index(
                drop=True
            )
        )

        targets.append(
            y_fold_valid.reset_index(
                drop=True
            )
        )

    meta_dataset = pd.concat(
        parts,
        ignore_index=True,
    )

    meta_target = pd.concat(
        targets,
        ignore_index=True,
    )

    return (
        meta_dataset,
        meta_target,
    )


# =============================================================================
# CREATE META FEATURES
# =============================================================================

def create_meta_features(
    pipelines,
    X,
    X_reference,
):

    reference = {}

    for feature in DRIFT_FEATURES:

        if feature not in X_reference.columns:
            continue

        values = pd.to_numeric(
            X_reference[feature],
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

    meta = pd.DataFrame(
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

        z = (
            values
            - reference[feature]["mean"]
        ) / reference[feature]["std"]

        z = z.fillna(0.0)

        meta[
            f"{feature}_drift_z"
        ] = z

        meta[
            f"{feature}_drift_abs"
        ] = z.abs()

    for model_name, pipeline in (
        pipelines.items()
    ):

        probability = (
            pipeline.predict_proba(
                X
            )[:, 1]
        )

        safe_name = (
            model_name
            .replace(
                " ",
                "_",
            )
            .replace(
                "-",
                "_",
            )
        )

        meta[
            f"{safe_name}_prob"
        ] = probability

        for feature in DRIFT_FEATURES:

            drift_column = (
                f"{feature}_drift_abs"
            )

            if drift_column in meta.columns:

                meta[
                    f"{safe_name}_x_"
                    f"{feature}_drift_abs"
                ] = (
                    probability
                    * meta[
                        drift_column
                    ]
                )

    return meta


# =============================================================================
# THRESHOLD SELECTION
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
# MAIN
# =============================================================================

def main():

    print("=" * 75)
    print(
        "SELF-HEALING ML — RECOVERY EXPERIMENT"
    )
    print("=" * 75)

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    X, y = load_dataset()

    print(
        f"\nDataset: {len(X)} records"
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
        f"\nTraining records:   {len(X_train)}"
    )

    print(
        f"Adaptation records: {len(X_adaptation)}"
    )

    print(
        f"Evaluation records: {len(X_eval)}"
    )

    # -------------------------------------------------------------------------
    # Create drift
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
        "\nControlled drift introduced:"
    )

    print(
        "  tenure × 1.5, capped at 72"
    )

    print(
        "  MonthlyCharges × 1.2, capped at 120"
    )

    # -------------------------------------------------------------------------
    # Detect drift
    # -------------------------------------------------------------------------

    drift_table = calculate_psi_table(
        X_train,
        X_adaptation_drifted,
    )

    drift_table = drift_table.sort_values(
        "PSI",
        ascending=False,
    )

    drift_table.to_csv(
        REPORT_DIR
        / "recovery_drift_report.csv",
        index=False,
    )

    max_psi = float(
        drift_table["PSI"].max()
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "DRIFT DETECTION"
    )

    print(
        "=" * 75
    )

    print(
        drift_table.head(10).to_string(
            index=False
        )
    )

    print(
        f"\nMaximum PSI: {max_psi:.6f}"
    )

    recovery_triggered = (
        max_psi
        >= PSI_ALERT_THRESHOLD
    )

    print(
        f"Recovery threshold: "
        f"{PSI_ALERT_THRESHOLD:.2f}"
    )

    print(
        "Recovery triggered:"
        f" {recovery_triggered}"
    )

    # -------------------------------------------------------------------------
    # Train conventional models BEFORE recovery
    # -------------------------------------------------------------------------

    models = build_models()

    static_pipelines = {}

    print(
        "\n"
        + "=" * 75
    )

    print(
        "TRAINING STATIC CONVENTIONAL MODELS"
    )

    print(
        "=" * 75
    )

    for model_name, model in (
        models.items()
    ):

        print(
            f"Training {model_name}..."
        )

        static_pipelines[
            model_name
        ] = train_model(
            model,
            X_train,
            y_train,
        )

    # -------------------------------------------------------------------------
    # Logistic Regression before recovery
    # -------------------------------------------------------------------------

    baseline_probability_before = (
        static_pipelines[
            "Logistic Regression"
        ]
        .predict_proba(
            X_eval_drifted
        )[:, 1]
    )

    baseline_before = calculate_metrics(
        y_eval,
        baseline_probability_before,
        threshold=0.50,
    )

    print(
        "\nStatic Logistic Regression "
        "before recovery:"
    )

    print(
        baseline_before
    )

    # -------------------------------------------------------------------------
    # Build SH-DAS before recovery
    # -------------------------------------------------------------------------

    print(
        "\n"
        + "=" * 75
    )

    print(
        "BUILDING SH-DAS BEFORE RECOVERY"
    )

    print(
        "=" * 75
    )

    # OOF meta data is created from the original training data.
    oof_meta, oof_target = (
        build_oof_meta_dataset(
            X_train,
            y_train,
        )
    )

    # Train meta-model.
    sh_das_meta = Pipeline(
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
    )

    sh_das_meta.fit(
        oof_meta,
        oof_target,
    )

    # Adaptation meta features.
    adaptation_meta = create_meta_features(
        static_pipelines,
        X_adaptation_drifted,
        X_train,
    )

    selected_threshold, threshold_table = (
        select_threshold(
            sh_das_meta,
            adaptation_meta,
            y_adaptation,
        )
    )

    threshold_table.to_csv(
        REPORT_DIR
        / "recovery_threshold_search.csv",
        index=False,
    )

    # Evaluation meta features.
    eval_meta_before = create_meta_features(
        static_pipelines,
        X_eval_drifted,
        X_train,
    )

    sh_das_probability_before = (
        sh_das_meta.predict_proba(
            eval_meta_before
        )[:, 1]
    )

    sh_das_before = calculate_metrics(
        y_eval,
        sh_das_probability_before,
        threshold=selected_threshold,
    )

    print(
        "\nSH-DAS before recovery:"
    )

    print(
        sh_das_before
    )

    # -------------------------------------------------------------------------
    # RECOVERY
    # -------------------------------------------------------------------------

    print(
        "\n"
        + "=" * 75
    )

    print(
        "SELF-HEALING RECOVERY"
    )

    print(
        "=" * 75
    )

    if recovery_triggered:

        print(
            "High drift detected."
        )

        print(
            "Adapting base models using "
            "recent drifted adaptation data..."
        )

        # Recovery dataset:
        # historical training data + observed drifted adaptation data.
        X_recovery_train = pd.concat(
            [
                X_train,
                X_adaptation_drifted,
            ],
            ignore_index=True,
        )

        y_recovery_train = pd.concat(
            [
                y_train.reset_index(
                    drop=True
                ),
                y_adaptation.reset_index(
                    drop=True
                ),
            ],
            ignore_index=True,
        )

        recovered_pipelines = {}

        for model_name, model in (
            models.items()
        ):

            print(
                f"Retraining {model_name}..."
            )

            recovered_pipelines[
                model_name
            ] = train_model(
                model,
                X_recovery_train,
                y_recovery_train,
            )

        # Rebuild meta model using the adaptation-informed
        # predictions from the recovered base models.
        recovery_meta = create_meta_features(
            recovered_pipelines,
            X_adaptation_drifted,
            X_recovery_train,
        )

        recovery_meta_model = Pipeline(
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
        )

        # Train recovery meta model using adaptation predictions
        # and the known adaptation labels.
        recovery_meta_model.fit(
            recovery_meta,
            y_adaptation,
        )

        # Select a new threshold from adaptation data.
        recovery_threshold, recovery_threshold_table = (
            select_threshold(
                recovery_meta_model,
                recovery_meta,
                y_adaptation,
            )
        )

        recovery_threshold_table.to_csv(
            REPORT_DIR
            / "recovery_adaptive_threshold_search.csv",
            index=False,
        )

        # Final recovered SH-DAS features.
        recovered_eval_meta = create_meta_features(
            recovered_pipelines,
            X_eval_drifted,
            X_recovery_train,
        )

        recovered_probability = (
            recovery_meta_model.predict_proba(
                recovered_eval_meta
            )[:, 1]
        )

        sh_das_after = calculate_metrics(
            y_eval,
            recovered_probability,
            threshold=recovery_threshold,
        )

        # Recovered Logistic Regression.
        recovered_logistic_probability = (
            recovered_pipelines[
                "Logistic Regression"
            ]
            .predict_proba(
                X_eval_drifted
            )[:, 1]
        )

        logistic_after = calculate_metrics(
            y_eval,
            recovered_logistic_probability,
            threshold=0.50,
        )

    else:

        print(
            "Drift threshold was not crossed."
        )

        print(
            "Recovery was correctly skipped."
        )

        recovered_pipelines = static_pipelines

        sh_das_after = sh_das_before

        logistic_after = baseline_before

        recovery_threshold = selected_threshold

    # -------------------------------------------------------------------------
    # RESULTS
    # -------------------------------------------------------------------------

    print(
        "\n"
        + "=" * 75
    )

    print(
        "SELF-HEALING RECOVERY RESULTS"
    )

    print(
        "=" * 75
    )

    recovery_rows = [
        {
            "Model": "Logistic Regression",
            "Stage": "Before Recovery",
            **baseline_before,
        },
        {
            "Model": "Logistic Regression",
            "Stage": "After Recovery",
            **logistic_after,
        },
        {
            "Model": "SH-DAS",
            "Stage": "Before Recovery",
            **sh_das_before,
        },
        {
            "Model": "SH-DAS",
            "Stage": "After Recovery",
            **sh_das_after,
        },
    ]

    recovery_df = pd.DataFrame(
        recovery_rows
    )

    recovery_df.to_csv(
        REPORT_DIR
        / "recovery_results.csv",
        index=False,
    )

    print(
        recovery_df.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Recovery gains
    # -------------------------------------------------------------------------

    gain_rows = []

    for model_name in [
        "Logistic Regression",
        "SH-DAS",
    ]:

        before = recovery_df[
            (
                recovery_df["Model"]
                == model_name
            )
            &
            (
                recovery_df["Stage"]
                == "Before Recovery"
            )
        ].iloc[0]

        after = recovery_df[
            (
                recovery_df["Model"]
                == model_name
            )
            &
            (
                recovery_df["Stage"]
                == "After Recovery"
            )
        ].iloc[0]

        for metric in [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
            "ROC-AUC",
        ]:

            gain_rows.append(
                {
                    "Model": model_name,
                    "Metric": metric,
                    "Before_Recovery": (
                        before[metric]
                    ),
                    "After_Recovery": (
                        after[metric]
                    ),
                    "Recovery_Gain": (
                        after[metric]
                        - before[metric]
                    ),
                }
            )

    gains_df = pd.DataFrame(
        gain_rows
    )

    gains_df.to_csv(
        REPORT_DIR
        / "recovery_gains.csv",
        index=False,
    )
    # =============================================================================
    # PERFORMANCE-AWARE RECOVERY GATE
    # =============================================================================

    print(
        "\n"
        + "=" * 75
    )

    print(
        "PERFORMANCE-AWARE RECOVERY GATE"
    )

    print(
        "=" * 75
    )

    # Extract SH-DAS before/after recovery.
    sh_das_before_row = recovery_df[
        (recovery_df["Model"] == "SH-DAS")
        &
        (recovery_df["Stage"] == "Before Recovery")
    ].iloc[0]

    sh_das_after_row = recovery_df[
        (recovery_df["Model"] == "SH-DAS")
        &
        (recovery_df["Stage"] == "After Recovery")
    ].iloc[0]

    # Candidate must:
    # 1. Not reduce F1-score.
    # 2. Not reduce ROC-AUC.
    # 3. Improve at least one of Recall or F1.
    #
    # This prevents a recovery candidate from being promoted
    # simply because one metric improved.

    f1_pass = (
        sh_das_after_row["F1-score"]
        >= sh_das_before_row["F1-score"]
    )

    auc_pass = (
        sh_das_after_row["ROC-AUC"]
        >= sh_das_before_row["ROC-AUC"]
    )

    recall_improved = (
        sh_das_after_row["Recall"]
        > sh_das_before_row["Recall"]
    )

    f1_improved = (
        sh_das_after_row["F1-score"]
        > sh_das_before_row["F1-score"]
    )

    promotion_allowed = (
        f1_pass
        and auc_pass
        and (recall_improved or f1_improved)
    )

    if promotion_allowed:

        recovery_decision = (
            "PROMOTE_RECOVERED_SH_DAS"
        )

        selected_stage = (
            "After Recovery"
        )

        selected_reason = (
            "Recovered SH-DAS maintained F1-score "
            "and ROC-AUC and improved at least one "
            "monitored objective."
        )

    else:

        recovery_decision = (
            "RETAIN_PREVIOUS_SH_DAS"
        )

        selected_stage = (
            "Before Recovery"
        )

        selected_reason = (
            "Recovered SH-DAS did not satisfy the "
            "performance gate. The previous SH-DAS "
            "model is retained to prevent harmful "
            "model replacement."
        )

    print(
        f"F1 gate:       {'PASS' if f1_pass else 'FAIL'}"
    )

    print(
        f"ROC-AUC gate:  {'PASS' if auc_pass else 'FAIL'}"
    )

    print(
        f"Recall improved: "
        f"{'YES' if recall_improved else 'NO'}"
    )

    print(
        f"Recovery decision: "
        f"{recovery_decision}"
    )

    print(
        f"Selected stage: "
        f"{selected_stage}"
    )

    print(
        f"Reason: {selected_reason}"
    )

    # Save machine-readable decision.
    gate_summary = {
        "recovery_decision": recovery_decision,
        "selected_stage": selected_stage,
        "f1_gate": bool(f1_pass),
        "roc_auc_gate": bool(auc_pass),
        "recall_improved": bool(recall_improved),
        "f1_improved": bool(f1_improved),
        "before_recovery": {
            "Accuracy": float(
                sh_das_before_row["Accuracy"]
            ),
            "Precision": float(
                sh_das_before_row["Precision"]
            ),
            "Recall": float(
                sh_das_before_row["Recall"]
            ),
            "F1-score": float(
                sh_das_before_row["F1-score"]
            ),
            "ROC-AUC": float(
                sh_das_before_row["ROC-AUC"]
            ),
        },
        "after_recovery": {
            "Accuracy": float(
                sh_das_after_row["Accuracy"]
            ),
            "Precision": float(
                sh_das_after_row["Precision"]
            ),
            "Recall": float(
                sh_das_after_row["Recall"]
            ),
            "F1-score": float(
                sh_das_after_row["F1-score"]
            ),
            "ROC-AUC": float(
                sh_das_after_row["ROC-AUC"]
            ),
        },
        "reason": selected_reason,
    }

    with (
        REPORT_DIR
        / "recovery_gate_decision.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            gate_summary,
            file,
            indent=2,
        )
    # -------------------------------------------------------------------------
    # ROC-AUC recovery chart
    # -------------------------------------------------------------------------

    plot_df = recovery_df[
        recovery_df["Model"].isin(
            [
                "Logistic Regression",
                "SH-DAS",
            ]
        )
    ]

    pivot = plot_df.pivot(
        index="Model",
        columns="Stage",
        values="ROC-AUC",
    )

    pivot = pivot[
        [
            "Before Recovery",
            "After Recovery",
        ]
    ]

    plt.figure(
        figsize=(9, 6)
    )

    x = np.arange(
        len(pivot.index)
    )

    width = 0.35

    plt.bar(
        x - width / 2,
        pivot[
            "Before Recovery"
        ],
        width,
        label="Before Recovery",
    )

    plt.bar(
        x + width / 2,
        pivot[
            "After Recovery"
        ],
        width,
        label="After Recovery",
    )

    plt.xticks(
        x,
        pivot.index,
    )

    plt.ylabel(
        "ROC-AUC"
    )

    plt.xlabel(
        "Model"
    )

    plt.title(
        "Performance Before and After Self-Healing Recovery"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR
        / "recovery_roc_auc.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    summary = {
        "experiment": (
            "Self-Healing Recovery Experiment"
        ),
        "dataset": (
            "IBM Telco Customer Churn"
        ),
        "drift_features": (
            DRIFT_FEATURES
        ),
        "psi_threshold": (
            PSI_ALERT_THRESHOLD
        ),
        "maximum_psi": max_psi,
        "recovery_triggered": (
            recovery_triggered
        ),
        "selected_threshold_before_recovery": (
            selected_threshold
        ),
        "selected_threshold_after_recovery": (
            recovery_threshold
        ),
        "same_final_evaluation_rows": True,
        "evaluation_note": (
            "Before and after recovery results "
            "use the same held-out evaluation "
            "records. Recovery uses training data "
            "plus labeled drifted adaptation data. "
            "Final evaluation data are not used "
            "during recovery."
        ),
    }

    with (
        REPORT_DIR
        / "self_healing_recovery_summary.json"
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
        "\n"
        + "=" * 75
    )

    print(
        "SELF-HEALING RECOVERY EXPERIMENT COMPLETED"
    )

    print(
        "=" * 75
    )

    print(
        f"\nReports saved to:\n{REPORT_DIR}"
    )


if __name__ == "__main__":
    main()