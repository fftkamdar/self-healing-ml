from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# ================================================================
# PROJECT PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Telco-Customer-Churn.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "proposed_model_v9"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.25


# ================================================================
# DATA
# ================================================================

def load_data() -> pd.DataFrame:

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if "TotalCharges" in df.columns:

        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce",
        )

    if "Churn" not in df.columns:
        raise ValueError(
            "Target column 'Churn' was not found."
        )

    return df


def engineer_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    data = df.copy()

    data = data.drop(
        columns=["customerID"],
        errors="ignore",
    )

    if "TotalCharges" in data.columns:

        data["TotalCharges"] = pd.to_numeric(
            data["TotalCharges"],
            errors="coerce",
        )

    # ------------------------------------------------------------
    # Average monthly spend
    # ------------------------------------------------------------

    if {
        "TotalCharges",
        "tenure",
    }.issubset(data.columns):

        safe_tenure = data["tenure"].replace(
            0,
            np.nan,
        )

        data["AverageMonthlySpend"] = (
            data["TotalCharges"]
            / safe_tenure
        )

    # ------------------------------------------------------------
    # Number of subscribed services
    # ------------------------------------------------------------

    service_columns = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    available_services = [
        column
        for column in service_columns
        if column in data.columns
    ]

    if available_services:

        service_yes = (
            data[available_services]
            .astype(str)
            .apply(
                lambda column:
                column.str.lower().eq("yes")
            )
        )

        data["ServiceCount"] = (
            service_yes.sum(axis=1)
        )

    # ------------------------------------------------------------
    # Month-to-month indicator
    # ------------------------------------------------------------

    if "Contract" in data.columns:

        data["IsMonthToMonth"] = (
            data["Contract"]
            .astype(str)
            .str.lower()
            .eq("month-to-month")
            .astype(int)
        )

    # ------------------------------------------------------------
    # Family support indicator
    # ------------------------------------------------------------

    if {
        "Partner",
        "Dependents",
    }.issubset(data.columns):

        partner_yes = (
            data["Partner"]
            .astype(str)
            .str.lower()
            .eq("yes")
        )

        dependent_yes = (
            data["Dependents"]
            .astype(str)
            .str.lower()
            .eq("yes")
        )

        data["FamilySupportCount"] = (
            partner_yes.astype(int)
            + dependent_yes.astype(int)
        )

    return data


def target_to_binary(
    series: pd.Series,
) -> pd.Series:

    normalized = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "yes": 1,
        "y": 1,
        "true": 1,
        "1": 1,
        "no": 0,
        "n": 0,
        "false": 0,
        "0": 0,
    }

    result = normalized.map(mapping)

    if result.isna().any():

        raise ValueError(
            "Could not safely convert Churn to binary."
        )

    return result.astype(int)


# ================================================================
# PREPROCESSOR
# ================================================================

def make_preprocessor(
    X: pd.DataFrame,
) -> ColumnTransformer:

    numeric_columns = (
        X.select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    categorical_columns = [
        column
        for column in X.columns
        if column not in numeric_columns
    ]

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
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                numeric_columns,
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_columns,
            ),
        ]
    )


def make_pipeline(
    estimator,
    X: pd.DataFrame,
) -> Pipeline:

    return Pipeline(
        steps=[
            (
                "preprocessor",
                make_preprocessor(X),
            ),
            (
                "model",
                estimator,
            ),
        ]
    )


# ================================================================
# METRICS
# ================================================================

def metrics_from_probability(
    probabilities: np.ndarray,
    y_true: pd.Series,
    threshold: float,
) -> dict[str, float]:

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


# ================================================================
# MODEL DEFINITIONS
# ================================================================

def get_models() -> dict:

    return {

        "Logistic Regression": LogisticRegression(
            C=2.0,
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=3,
            subsample=0.90,
            random_state=RANDOM_STATE,
        ),

        "Support Vector Machine": SVC(
            C=2.0,
            kernel="rbf",
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "HistGradientBoosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),

        "AdaBoost": AdaBoostClassifier(
            n_estimators=250,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
        ),
    }


# ================================================================
# VALIDATION OBJECTIVE
# ================================================================

def validation_objective(
    metrics: dict[str, float],
    reference_maxima: dict[str, float],
) -> tuple:

    dominance_count = sum(
        metrics[m] >= reference_maxima[m]
        for m in reference_maxima
    )

    margins = [
        metrics[m] - reference_maxima[m]
        for m in reference_maxima
    ]

    mean_margin = float(
        np.mean(margins)
    )

    # Extra emphasis on F1 and ROC-AUC as secondary
    # tie-breakers.
    balanced_score = (
        0.15 * metrics["Accuracy"]
        + 0.15 * metrics["Precision"]
        + 0.15 * metrics["Recall"]
        + 0.35 * metrics["F1-score"]
        + 0.20 * metrics["ROC-AUC"]
    )

    return (
        dominance_count,
        mean_margin,
        balanced_score,
    )


# ================================================================
# RANDOM BLEND SEARCH
# ================================================================

def random_blend_search(
    validation_probabilities: dict[str, np.ndarray],
    y_validation: pd.Series,
    reference_maxima: dict[str, float],
    rng: np.random.Generator,
    n_samples: int = 5000,
):

    model_names = list(
        validation_probabilities.keys()
    )

    probability_matrix = np.column_stack(
        [
            validation_probabilities[name]
            for name in model_names
        ]
    )

    best = None

    thresholds = np.arange(
        0.20,
        0.81,
        0.01,
    )

    for _ in range(n_samples):

        weights = rng.dirichlet(
            np.ones(len(model_names))
        )

        blended = (
            probability_matrix
            @ weights
        )

        for threshold in thresholds:

            metrics = metrics_from_probability(
                blended,
                y_validation,
                float(threshold),
            )

            objective = validation_objective(
                metrics,
                reference_maxima,
            )

            if (
                best is None
                or objective > best["objective"]
            ):

                best = {
                    "objective": objective,
                    "weights": weights.copy(),
                    "threshold": float(threshold),
                    "metrics": metrics,
                    "model_names": model_names.copy(),
                }

    return best


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 90)
    print("SH-DAS V9 — MULTI-OBJECTIVE PERFORMANCE SEARCH")
    print("=" * 90)

    df = load_data()

    df = engineer_features(df)

    y = target_to_binary(
        df["Churn"]
    )

    X = df.drop(
        columns=["Churn"],
        errors="ignore",
    )

    # ------------------------------------------------------------
    # Fixed train / validation / test split
    # ------------------------------------------------------------

    (
        X_train_full,
        X_test,
        y_train_full,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    (
        X_train,
        X_validation,
        y_train,
        y_validation,
    ) = train_test_split(
        X_train_full,
        y_train_full,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )

    print()
    print(
        f"Train:      {len(X_train):,}"
    )
    print(
        f"Validation: {len(X_validation):,}"
    )
    print(
        f"Test:       {len(X_test):,}"
    )

    # ------------------------------------------------------------
    # Fit all candidate models
    # ------------------------------------------------------------

    models = get_models()

    validation_probabilities = {}
    test_probabilities = {}

    existing_results = []

    print()
    print("MODEL EVALUATION")
    print("-" * 90)

    for name, estimator in models.items():

        model = make_pipeline(
            estimator,
            X_train,
        )

        print(
            f"Training {name} ..."
        )

        started = time.perf_counter()

        model.fit(
            X_train,
            y_train,
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        val_prob = (
            model
            .predict_proba(X_validation)[:, 1]
        )

        test_prob = (
            model
            .predict_proba(X_test)[:, 1]
        )

        validation_probabilities[name] = (
            val_prob
        )

        test_probabilities[name] = (
            test_prob
        )

        print(
            f"{name:25s} "
            f"ValF1={metrics_from_probability(val_prob, y_validation, 0.50)['F1-score']:.4f} "
            f"ValAUC={metrics_from_probability(val_prob, y_validation, 0.50)['ROC-AUC']:.4f}"
        )

        # Existing five-model family.
        if name in {
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "Gradient Boosting",
            "Support Vector Machine",
        }:

            test_metrics = metrics_from_probability(
                test_prob,
                y_test,
                0.50,
            )

            existing_results.append(
                {
                    "Model": name,
                    "Type": "Existing",
                    "Accuracy": test_metrics["Accuracy"],
                    "Precision": test_metrics["Precision"],
                    "Recall": test_metrics["Recall"],
                    "F1-score": test_metrics["F1-score"],
                    "ROC-AUC": test_metrics["ROC-AUC"],
                    "Threshold": 0.50,
                    "Training_Time_Seconds": elapsed,
                }
            )

    existing_df = pd.DataFrame(
        existing_results
    )

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
    ]

    # ------------------------------------------------------------
    # IMPORTANT:
    # Reference maxima must be calculated on VALIDATION data,
    # not test data, during model selection.
    # ------------------------------------------------------------

    validation_existing_results = []

    for name in [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "Support Vector Machine",
    ]:

        metrics = metrics_from_probability(
            validation_probabilities[name],
            y_validation,
            0.50,
        )

        validation_existing_results.append(
            metrics
        )

    validation_existing_df = pd.DataFrame(
        validation_existing_results
    )

    validation_reference_maxima = {
        metric: float(
            validation_existing_df[metric].max()
        )
        for metric in metric_columns
    }

    test_reference_maxima = {
        metric: float(
            existing_df[metric].max()
        )
        for metric in metric_columns
    }

    # ------------------------------------------------------------
    # Search thresholds for each individual candidate.
    # ------------------------------------------------------------

    best_individual = None

    for name, probabilities in (
        validation_probabilities.items()
    ):

        for threshold in np.arange(
            0.20,
            0.81,
            0.01,
        ):

            metrics = metrics_from_probability(
                probabilities,
                y_validation,
                float(threshold),
            )

            objective = validation_objective(
                metrics,
                validation_reference_maxima,
            )

            if (
                best_individual is None
                or objective
                > best_individual["objective"]
            ):

                best_individual = {
                    "name": name,
                    "objective": objective,
                    "threshold": float(threshold),
                    "metrics": metrics,
                }

    # ------------------------------------------------------------
    # Randomized probability blend.
    # ------------------------------------------------------------

    print()
    print("SEARCHING PROBABILITY BLENDS")
    print("-" * 90)

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    blend_best = random_blend_search(
        validation_probabilities,
        y_validation,
        validation_reference_maxima,
        rng,
        n_samples=20,
    )

    candidates = [
        (
            "Best Individual V9",
            best_individual["metrics"],
            best_individual["threshold"],
            None,
        ),
        (
            "SH-DAS V9 Blend",
            blend_best["metrics"],
            blend_best["threshold"],
            blend_best,
        ),
    ]

    candidates.sort(
        key=lambda item: validation_objective(
            item[1],
            validation_reference_maxima,
        ),
        reverse=True,
    )

    selected_name, selected_validation_metrics, selected_threshold, selected_blend = candidates[0]

    # ------------------------------------------------------------
    # Test evaluation
    # ------------------------------------------------------------

    if selected_blend is None:

        selected_probabilities = (
            validation_probabilities[
                best_individual["name"]
            ]
        )

        test_probabilities_selected = (
            test_probabilities[
                best_individual["name"]
            ]
        )

        selected_model_description = (
            best_individual["name"]
        )

        weights_dict = {
            best_individual["name"]: 1.0
        }

    else:

        model_names = (
            selected_blend["model_names"]
        )

        weights = (
            selected_blend["weights"]
        )

        validation_matrix = np.column_stack(
            [
                validation_probabilities[name]
                for name in model_names
            ]
        )

        test_matrix = np.column_stack(
            [
                test_probabilities[name]
                for name in model_names
            ]
        )

        selected_probabilities = (
            validation_matrix @ weights
        )

        test_probabilities_selected = (
            test_matrix @ weights
        )

        weights_dict = {
            name: float(weight)
            for name, weight
            in zip(model_names, weights)
            if weight > 0.01
        }

        selected_model_description = (
            "Probability blend of "
            + ", ".join(weights_dict.keys())
        )

    selected_test_metrics = metrics_from_probability(
        test_probabilities_selected,
        y_test,
        selected_threshold,
    )

    # ------------------------------------------------------------
    # Test dominance check
    # ------------------------------------------------------------

    test_dominance = {
        metric: (
            selected_test_metrics[metric]
            > test_reference_maxima[metric]
        )
        for metric in metric_columns
    }

    dominance_count = sum(
        test_dominance.values()
    )

    all_metrics_pass = all(
        test_dominance.values()
    )

    # ------------------------------------------------------------
    # Final comparison table
    # ------------------------------------------------------------

    final_rows = existing_results.copy()

    final_rows.append(
        {
            "Model": "SH-DAS V9 (Proposed)",
            "Type": "Proposed",
            "Accuracy": selected_test_metrics["Accuracy"],
            "Precision": selected_test_metrics["Precision"],
            "Recall": selected_test_metrics["Recall"],
            "F1-score": selected_test_metrics["F1-score"],
            "ROC-AUC": selected_test_metrics["ROC-AUC"],
            "Threshold": selected_threshold,
            "Training_Time_Seconds": np.nan,
        }
    )

    final_df = pd.DataFrame(
        final_rows
    )

    comparison_path = (
        OUTPUT_DIR
        / "v9_final_comparison.csv"
    )

    final_df.to_csv(
        comparison_path,
        index=False,
    )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    summary = {
        "model": "SH-DAS V9",
        "selected_model_description":
            selected_model_description,
        "threshold":
            selected_threshold,
        "weights":
            weights_dict,
        "validation_metrics":
            selected_validation_metrics,
        "test_metrics":
            selected_test_metrics,
        "validation_reference_maxima":
            validation_reference_maxima,
        "test_reference_maxima":
            test_reference_maxima,
        "test_dominance":
            test_dominance,
        "dominance_count":
            dominance_count,
        "all_metrics_beat_all_five":
            all_metrics_pass,
        "selection_protocol":
            "Model and threshold selected using training and validation data only. Held-out test set evaluated once for final comparison.",
        "feature_engineering":
            [
                "AverageMonthlySpend",
                "ServiceCount",
                "IsMonthToMonth",
                "FamilySupportCount",
            ],
    }

    summary_path = (
        OUTPUT_DIR
        / "v9_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    # ------------------------------------------------------------
    # Save readable report
    # ------------------------------------------------------------

    report_path = (
        OUTPUT_DIR
        / "v9_report.txt"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "SH-DAS V9 — MULTI-OBJECTIVE PERFORMANCE SEARCH\n"
        )

        file.write(
            "=" * 90
            + "\n\n"
        )

        file.write(
            f"Selected approach: "
            f"{selected_model_description}\n"
        )

        file.write(
            f"Threshold: "
            f"{selected_threshold:.2f}\n\n"
        )

        file.write(
            "FINAL TEST METRICS\n"
        )

        for metric in metric_columns:

            file.write(
                f"{metric:12s}: "
                f"{selected_test_metrics[metric]:.6f}\n"
            )

        file.write(
            "\nBEST EXISTING TEST VALUES\n"
        )

        for metric in metric_columns:

            file.write(
                f"{metric:12s}: "
                f"{test_reference_maxima[metric]:.6f}\n"
            )

        file.write(
            "\nDOMINANCE CHECK\n"
        )

        for metric in metric_columns:

            file.write(
                f"{metric:12s}: "
                f"{'YES' if test_dominance[metric] else 'NO'}\n"
            )

        file.write(
            "\nDOMINANCE COUNT: "
            f"{dominance_count}/5\n"
        )

        file.write(
            "ALL FIVE METRICS BEAT ALL FIVE MODELS: "
            f"{'YES' if all_metrics_pass else 'NO'}\n"
        )

        file.write(
            "\nBLEND WEIGHTS\n"
        )

        for name, weight in weights_dict.items():

            file.write(
                f"{name}: {weight:.4f}\n"
            )

    # ------------------------------------------------------------
    # Save model bundle if the selected approach is a blend.
    # ------------------------------------------------------------

    model_bundle = {
        "approach":
            selected_model_description,
        "threshold":
            selected_threshold,
        "weights":
            weights_dict,
    }

    bundle_path = (
        OUTPUT_DIR
        / "sh_das_v9_model_bundle.joblib"
    )

    joblib.dump(
        model_bundle,
        bundle_path,
    )

    # ------------------------------------------------------------
    # Console result
    # ------------------------------------------------------------

    print()
    print("=" * 90)
    print("FINAL V9 RESULT")
    print("=" * 90)

    print(
        f"Selected approach: "
        f"{selected_model_description}"
    )

    print(
        f"Threshold: "
        f"{selected_threshold:.2f}"
    )

    print()

    for metric in metric_columns:

        print(
            f"{metric:12s}: "
            f"{selected_test_metrics[metric]:.4f} "
            f"(existing max: "
            f"{test_reference_maxima[metric]:.4f}) "
            f"{'?' if test_dominance[metric] else '?'}"
        )

    print()
    print(
        f"DOMINANCE COUNT: "
        f"{dominance_count}/5"
    )

    print(
        "BEATS ALL FIVE ON ALL FIVE METRICS:",
        "YES" if all_metrics_pass else "NO",
    )

    print()
    print(
        f"Saved: {comparison_path}"
    )

    print(
        f"Saved: {summary_path}"
    )

    print(
        f"Saved: {report_path}"
    )

    print(
        f"Saved: {bundle_path}"
    )


if __name__ == "__main__":
    main()


