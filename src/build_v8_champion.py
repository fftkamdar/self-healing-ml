from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
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


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"

OUTPUT_DIR = PROJECT_ROOT / "reports" / "proposed_model_v8"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.25


# ---------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce",
        )

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
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

    # Additional deterministic features.
    if {"TotalCharges", "tenure"}.issubset(data.columns):
        safe_tenure = data["tenure"].replace(0, np.nan)
        data["AverageMonthlySpend"] = (
            data["TotalCharges"] / safe_tenure
        )

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
                lambda column: column.str.lower().eq("yes")
            )
        )
        data["ServiceCount"] = service_yes.sum(axis=1)

    if "Contract" in data.columns:
        data["IsMonthToMonth"] = (
            data["Contract"]
            .astype(str)
            .str.lower()
            .eq("month-to-month")
            .astype(int)
        )

    if {"Partner", "Dependents"}.issubset(data.columns):
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


def target_to_binary(series: pd.Series) -> pd.Series:
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
            "Could not safely convert target to binary."
        )

    return result.astype(int)


# ---------------------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------------------

def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = [
        column
        for column in X.columns
        if column not in numeric_columns
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
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
                SimpleImputer(strategy="most_frequent"),
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
        ],
        remainder="drop",
    )


def make_pipeline(estimator, X: pd.DataFrame) -> Pipeline:
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


# ---------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------

def evaluate(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.50,
) -> dict[str, float]:

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "Accuracy": float(
            accuracy_score(y, predictions)
        ),
        "Precision": float(
            precision_score(
                y,
                predictions,
                zero_division=0,
            )
        ),
        "Recall": float(
            recall_score(
                y,
                predictions,
                zero_division=0,
            )
        ),
        "F1-score": float(
            f1_score(
                y,
                predictions,
                zero_division=0,
            )
        ),
        "ROC-AUC": float(
            roc_auc_score(
                y,
                probabilities,
            )
        ),
    }


# ---------------------------------------------------------------------
# THRESHOLD SEARCH
# ---------------------------------------------------------------------

def choose_threshold(
    model,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> tuple[float, dict[str, float]]:

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    best_threshold = 0.50
    best_metrics = evaluate(
        model,
        X_validation,
        y_validation,
        threshold=0.50,
    )

    best_score = -1.0

    for threshold in np.arange(
        0.25,
        0.71,
        0.01,
    ):

        predictions = (
            probabilities >= threshold
        ).astype(int)

        accuracy = accuracy_score(
            y_validation,
            predictions,
        )

        precision = precision_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        # Balanced validation objective.
        score = (
            0.20 * accuracy
            + 0.20 * precision
            + 0.20 * recall
            + 0.35 * f1
            + 0.05 * 0.5
        )

        if score > best_score:

            best_score = score
            best_threshold = float(threshold)

            best_metrics = {
                "Accuracy": float(accuracy),
                "Precision": float(precision),
                "Recall": float(recall),
                "F1-score": float(f1),
                "ROC-AUC": float(
                    roc_auc_score(
                        y_validation,
                        probabilities,
                    )
                ),
            }

    return (
        best_threshold,
        best_metrics,
    )


# ---------------------------------------------------------------------
# CANDIDATE MODELS
# ---------------------------------------------------------------------

def candidate_definitions():

    candidates = []

    candidates.append(
        (
            "LR_Tuned",
            LogisticRegression(
                C=2.0,
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        )
    )

    candidates.append(
        (
            "RandomForest_Tuned",
            RandomForestClassifier(
                n_estimators=500,
                max_depth=10,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        )
    )

    candidates.append(
        (
            "ExtraTrees_Tuned",
            ExtraTreesClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        )
    )

    candidates.append(
        (
            "ExtraTrees_Wide",
            ExtraTreesClassifier(
                n_estimators=500,
                max_depth=12,
                min_samples_leaf=1,
                max_features=0.70,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        )
    )

    candidates.append(
        (
            "GradientBoosting_Tuned",
            GradientBoostingClassifier(
                n_estimators=250,
                learning_rate=0.05,
                max_depth=3,
                min_samples_leaf=3,
                subsample=0.90,
                random_state=RANDOM_STATE,
            ),
        )
    )

    candidates.append(
        (
            "SVM_Tuned",
            SVC(
                C=2.0,
                kernel="rbf",
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        )
    )

    return candidates


# ---------------------------------------------------------------------
# MAIN EXPERIMENT
# ---------------------------------------------------------------------

def main():

    print("=" * 90)
    print("SH-DAS V8 — PERFORMANCE SEARCH")
    print("=" * 90)

    df = load_data()
    df = engineer_features(df)

    y = target_to_binary(df["Churn"])

    X = df.drop(
        columns=["Churn"],
        errors="ignore",
    )

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

    results = []

    fitted_models = {}

    # --------------------------------------------------------------
    # Existing five-model family on THIS SAME split
    # --------------------------------------------------------------

    baseline_estimators = [
        (
            "Logistic Regression",
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "Decision Tree",
            __import__(
                "sklearn.tree",
                fromlist=["DecisionTreeClassifier"],
            ).DecisionTreeClassifier(
                max_depth=6,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=400,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        (
            "Gradient Boosting",
            GradientBoostingClassifier(
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "Support Vector Machine",
            SVC(
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ]

    print()
    print("BASELINE FIVE")
    print("-" * 90)

    for name, estimator in baseline_estimators:

        model = make_pipeline(
            estimator,
            X_train_full,
        )

        started = time.perf_counter()

        model.fit(
            X_train_full,
            y_train_full,
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        metrics = evaluate(
            model,
            X_test,
            y_test,
            threshold=0.50,
        )

        results.append(
            {
                "Model": name,
                "Type": "Existing",
                "Accuracy": metrics["Accuracy"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1-score": metrics["F1-score"],
                "ROC-AUC": metrics["ROC-AUC"],
                "Threshold": 0.50,
                "Training_Time_Seconds": elapsed,
            }
        )

        print(
            f"{name:25s} "
            f"Acc={metrics['Accuracy']:.4f} "
            f"Prec={metrics['Precision']:.4f} "
            f"Rec={metrics['Recall']:.4f} "
            f"F1={metrics['F1-score']:.4f} "
            f"AUC={metrics['ROC-AUC']:.4f}"
        )

    # --------------------------------------------------------------
    # Candidate search
    # --------------------------------------------------------------

    print()
    print("V8 CANDIDATES")
    print("-" * 90)

    candidate_scores = []

    for name, estimator in candidate_definitions():

        model = make_pipeline(
            estimator,
            X_train,
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

        threshold, validation_metrics = (
            choose_threshold(
                model,
                X_validation,
                y_validation,
            )
        )

        test_metrics = evaluate(
            model,
            X_test,
            y_test,
            threshold=threshold,
        )

        candidate_scores.append(
            (
                name,
                model,
                threshold,
                validation_metrics,
                test_metrics,
                elapsed,
            )
        )

        print(
            f"{name:25s} "
            f"ValF1={validation_metrics['F1-score']:.4f} "
            f"ValAUC={validation_metrics['ROC-AUC']:.4f} "
            f"TestAcc={test_metrics['Accuracy']:.4f} "
            f"TestPrec={test_metrics['Precision']:.4f} "
            f"TestRec={test_metrics['Recall']:.4f} "
            f"TestF1={test_metrics['F1-score']:.4f} "
            f"TestAUC={test_metrics['ROC-AUC']:.4f}"
        )

    # --------------------------------------------------------------
    # Build stacking model from strong candidates
    # --------------------------------------------------------------

    selected_base = []

    for name, estimator in [
        (
            "lr",
            LogisticRegression(
                C=2.0,
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=350,
                max_depth=10,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        (
            "et",
            ExtraTreesClassifier(
                n_estimators=350,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        (
            "gb",
            GradientBoostingClassifier(
                n_estimators=250,
                learning_rate=0.05,
                max_depth=3,
                min_samples_leaf=3,
                random_state=RANDOM_STATE,
            ),
        ),
    ]:

        selected_base.append(
            (
                name,
                make_pipeline(
                    estimator,
                    X_train,
                ),
            )
        )

    print()
    print("STACKING MODEL")
    print("-" * 90)

    stacking_model = StackingClassifier(
        estimators=selected_base,
        final_estimator=LogisticRegression(
            C=1.0,
            max_iter=3000,
            class_weight="balanced",
        ),
        stack_method="predict_proba",
        passthrough=False,
        cv=5,
        n_jobs=-1,
    )

    started = time.perf_counter()

    stacking_model.fit(
        X_train,
        y_train,
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    stacking_threshold, stacking_validation = (
        choose_threshold(
            stacking_model,
            X_validation,
            y_validation,
        )
    )

    stacking_test = evaluate(
        stacking_model,
        X_test,
        y_test,
        threshold=stacking_threshold,
    )

    candidate_scores.append(
        (
            "SH-DAS V8 Stacking",
            stacking_model,
            stacking_threshold,
            stacking_validation,
            stacking_test,
            elapsed,
        )
    )

    print(
        f"SH-DAS V8 Stacking       "
        f"ValF1={stacking_validation['F1-score']:.4f} "
        f"ValAUC={stacking_validation['ROC-AUC']:.4f} "
        f"TestAcc={stacking_test['Accuracy']:.4f} "
        f"TestPrec={stacking_test['Precision']:.4f} "
        f"TestRec={stacking_test['Recall']:.4f} "
        f"TestF1={stacking_test['F1-score']:.4f} "
        f"TestAUC={stacking_test['ROC-AUC']:.4f}"
    )

    # --------------------------------------------------------------
    # Reference maxima from the five existing models
    # --------------------------------------------------------------

    existing_df = pd.DataFrame(
        results
    )

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
    ]

    maxima = {
        metric: float(
            existing_df[metric].max()
        )
        for metric in metric_columns
    }

    # --------------------------------------------------------------
    # Identify candidate with strongest overall validation score.
    # --------------------------------------------------------------

    def validation_score(item):

        (
            _name,
            _model,
            _threshold,
            validation_metrics,
            _test_metrics,
            _elapsed,
        ) = item

        return (
            0.20 * validation_metrics["Accuracy"]
            + 0.20 * validation_metrics["Precision"]
            + 0.20 * validation_metrics["Recall"]
            + 0.30 * validation_metrics["F1-score"]
            + 0.10 * validation_metrics["ROC-AUC"]
        )

    candidate_scores.sort(
        key=validation_score,
        reverse=True,
    )

    best_name, best_model, best_threshold, best_validation, best_test, best_elapsed = candidate_scores[0]

    # --------------------------------------------------------------
    # Final comparison table
    # --------------------------------------------------------------

    final_rows = list(results)

    final_rows.append(
        {
            "Model": "SH-DAS V8 (Proposed)",
            "Type": "Proposed",
            "Accuracy": best_test["Accuracy"],
            "Precision": best_test["Precision"],
            "Recall": best_test["Recall"],
            "F1-score": best_test["F1-score"],
            "ROC-AUC": best_test["ROC-AUC"],
            "Threshold": best_threshold,
            "Training_Time_Seconds": best_elapsed,
        }
    )

    final_df = pd.DataFrame(
        final_rows
    )

    comparison_path = (
        OUTPUT_DIR
        / "v8_final_comparison.csv"
    )

    final_df.to_csv(
        comparison_path,
        index=False,
    )

    # --------------------------------------------------------------
    # Does V8 beat every one of the five?
    # --------------------------------------------------------------

    proposed_beats_all = {}

    for metric in metric_columns:

        proposed_beats_all[metric] = (
            best_test[metric]
            > maxima[metric]
        )

    all_metrics_pass = all(
        proposed_beats_all.values()
    )

    summary = {
        "model": "SH-DAS V8",
        "selected_candidate": best_name,
        "threshold": best_threshold,
        "validation_metrics": best_validation,
        "test_metrics": best_test,
        "existing_model_maxima": maxima,
        "beats_all_existing_models": proposed_beats_all,
        "all_metrics_beat_all_five": all_metrics_pass,
        "experiment_note": (
            "V8 candidate selection used training and validation data only. "
            "The final comparison uses the untouched held-out test set."
        ),
    }

    summary_path = (
        OUTPUT_DIR
        / "v8_summary.json"
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

    # Save candidate model separately.
    model_path = (
        OUTPUT_DIR
        / "sh_das_v8_candidate.joblib"
    )

    joblib.dump(
        best_model,
        model_path,
    )

    # Human-readable report.
    report_path = (
        OUTPUT_DIR
        / "v8_report.txt"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "SH-DAS V8 PERFORMANCE SEARCH\n"
        )
        file.write(
            "=" * 90
            + "\n\n"
        )

        file.write(
            f"Selected candidate: {best_name}\n"
        )
        file.write(
            f"Threshold: {best_threshold:.2f}\n\n"
        )

        file.write(
            "Existing-model maxima on the same held-out test set:\n"
        )

        for metric, value in maxima.items():
            file.write(
                f"  {metric}: {value:.6f}\n"
            )

        file.write(
            "\nV8 test performance:\n"
        )

        for metric, value in best_test.items():
            file.write(
                f"  {metric}: {value:.6f}\n"
            )

        file.write(
            "\nBeat all five existing models:\n"
        )

        for metric, passed in proposed_beats_all.items():
            file.write(
                f"  {metric}: "
                f"{'YES' if passed else 'NO'}\n"
            )

        file.write(
            "\nALL FIVE METRICS PASSED: "
            f"{'YES' if all_metrics_pass else 'NO'}\n"
        )

    print()
    print("=" * 90)
    print("FINAL V8 RESULT")
    print("=" * 90)

    print(
        f"Selected candidate: {best_name}"
    )

    print(
        f"Threshold: {best_threshold:.2f}"
    )

    for metric, value in best_test.items():
        print(
            f"{metric:12s}: {value:.4f} "
            f"(existing max: {maxima[metric]:.4f})"
        )

    print()
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
        f"Saved: {model_path}"
    )


if __name__ == "__main__":
    main()

