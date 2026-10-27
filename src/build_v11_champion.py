from pathlib import Path
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
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

warnings.filterwarnings("ignore")

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "proposed_model_v11"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE = {
    "Accuracy": 0.8013,
    "Precision": 0.6679,
    "Recall": 0.7968,
    "F1-score": 0.6179,
    "ROC-AUC": 0.8430,
}

DATA_CANDIDATES = [
    ROOT / "data" / "raw" / "Telco-Customer-Churn.csv",
    ROOT / "data" / "Telco-Customer-Churn.csv",
    ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    ROOT / "Telco-Customer-Churn.csv",
]

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


def find_dataset():
    for path in DATA_CANDIDATES:
        if path.exists():
            return path

    for path in ROOT.rglob("*.csv"):
        name = path.name.lower()
        if "telco" in name or "churn" in name:
            return path

    raise FileNotFoundError(
        "Could not find Telco Customer Churn CSV."
    )


def engineer_features(df):
    df = df.copy()

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

    if "tenure" in df.columns and "TotalCharges" in df.columns:
        df["AverageMonthlySpend"] = (
            df["TotalCharges"]
            / df["tenure"].replace(0, np.nan)
        )

    service_columns = [
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    existing_services = [
        c for c in service_columns if c in df.columns
    ]

    if existing_services:
        df["ServiceCount"] = 0

        for col in existing_services:
            df["ServiceCount"] += (
                ~df[col]
                .astype(str)
                .isin(
                    [
                        "No",
                        "No internet service",
                        "nan",
                    ]
                )
            ).astype(int)

    if "Contract" in df.columns:
        df["IsMonthToMonth"] = (
            df["Contract"]
            .astype(str)
            .eq("Month-to-month")
            .astype(int)
        )

    family_columns = ["Partner", "Dependents"]

    existing_family = [
        c for c in family_columns if c in df.columns
    ]

    if existing_family:
        df["FamilySupportCount"] = 0

        for col in existing_family:
            df["FamilySupportCount"] += (
                df[col]
                .astype(str)
                .eq("Yes")
            ).astype(int)

    if "tenure" in df.columns and "MonthlyCharges" in df.columns:
        df["TenureChargeInteraction"] = (
            df["tenure"] * df["MonthlyCharges"]
        )

    return df


def prepare_data(df):
    if "Churn" not in df.columns:
        raise ValueError("Churn column not found.")

    y = (
        df["Churn"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
    )

    if y.isna().any():
        raise ValueError(
            "Unexpected values found in Churn."
        )

    X = df.drop(columns=["Churn"])

    return X, y.astype(int)


def make_preprocessor(X):
    numeric_cols = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_cols = [
        c for c in X.columns
        if c not in numeric_cols
    ]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
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
                numeric_pipe,
                numeric_cols,
            ),
            (
                "cat",
                categorical_pipe,
                categorical_cols,
            ),
        ],
        remainder="drop",
    )


def build_estimators(y_train):
    estimators = [
        (
            "lr",
            LogisticRegression(
                C=0.70,
                max_iter=4000,
                class_weight="balanced",
                random_state=SEED,
            ),
        ),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=400,
                max_depth=9,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=SEED,
                n_jobs=1,
            ),
        ),
        (
            "extra",
            ExtraTreesClassifier(
                n_estimators=400,
                max_depth=12,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced",
                random_state=SEED,
                n_jobs=1,
            ),
        ),
        (
            "gb",
            GradientBoostingClassifier(
                n_estimators=260,
                learning_rate=0.035,
                max_depth=2,
                min_samples_leaf=8,
                subsample=0.90,
                random_state=SEED,
            ),
        ),
        (
            "hgb",
            HistGradientBoostingClassifier(
                max_iter=250,
                learning_rate=0.035,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                l2_regularization=1.5,
                random_state=SEED,
            ),
        ),
        (
            "svm",
            SVC(
                C=1.20,
                kernel="rbf",
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=SEED,
            ),
        ),
    ]

    if HAS_XGB:
        positive = int(y_train.sum())
        negative = int(len(y_train) - positive)

        scale_pos_weight = (
            negative / positive
            if positive > 0
            else 1.0
        )

        estimators.append(
            (
                "xgb",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=3,
                    learning_rate=0.03,
                    min_child_weight=5,
                    subsample=0.90,
                    colsample_bytree=0.80,
                    reg_lambda=5.0,
                    reg_alpha=0.10,
                    scale_pos_weight=scale_pos_weight,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=SEED,
                    n_jobs=1,
                ),
            )
        )

    return estimators


def metrics_from_probability(
    y_true,
    probability,
    threshold,
):
    prediction = (
        probability >= threshold
    ).astype(int)

    return {
        "Accuracy": float(
            accuracy_score(
                y_true,
                prediction,
            )
        ),
        "Precision": float(
            precision_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "Recall": float(
            recall_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "F1-score": float(
            f1_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "ROC-AUC": float(
            roc_auc_score(
                y_true,
                probability,
            )
        ),
    }


def objective_score(metrics):
    metric_names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
    ]

    ratios = [
        metrics[m] / REFERENCE[m]
        for m in metric_names
    ]

    all_above = all(
        value > 1.0
        for value in ratios
    )

    return (
        int(all_above),
        float(min(ratios)),
        float(np.mean(ratios)),
        float(metrics["F1-score"]),
        float(metrics["ROC-AUC"]),
    )


def print_metrics(title, metrics):
    print("\n" + title)
    print("-" * 90)

    for metric in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
    ]:
        actual = metrics[metric]
        reference = REFERENCE[metric]
        symbol = "PASS" if actual > reference else "MISS"

        print(
            f"{metric:<12}: "
            f"{actual:.4f} "
            f"(reference: {reference:.4f}) "
            f"{symbol}"
        )


def main():
    start = time.time()

    print("=" * 90)
    print("SH-DAS V11 - STABLE STACKING AND PARETO THRESHOLD SEARCH")
    print("=" * 90)

    dataset = find_dataset()

    print(f"\nDataset: {dataset}")
    print(
        f"XGBoost available: "
        f"{'YES' if HAS_XGB else 'NO'}"
    )

    df = pd.read_csv(dataset)
    df = engineer_features(df)

    X, y = prepare_data(df)

    # Reproduce the established 60/20/20 split exactly.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        stratify=y,
        random_state=SEED,
    )

    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=SEED,
    )

    print(f"\nTrain:      {len(X_train):,}")
    print(f"Validation: {len(X_validation):,}")
    print(f"Test:       {len(X_test):,}")

    engineered = [
        c for c in [
            "AverageMonthlySpend",
            "ServiceCount",
            "IsMonthToMonth",
            "FamilySupportCount",
            "TenureChargeInteraction",
        ]
        if c in X.columns
    ]

    print("\nEngineered features:")
    for item in engineered:
        print(f"  + {item}")

    preprocessor = make_preprocessor(X_train)

    X_train_encoded = preprocessor.fit_transform(
        X_train
    )

    X_validation_encoded = preprocessor.transform(
        X_validation
    )

    estimators = build_estimators(y_train)

    print("\nBASE MODELS")
    print("-" * 90)

    for name, _ in estimators:
        print(f"  + {name}")

    final_meta = LogisticRegression(
        C=0.25,
        max_iter=4000,
        class_weight="balanced",
        random_state=SEED,
    )

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=final_meta,
        cv=4,
        stack_method="predict_proba",
        passthrough=True,
        n_jobs=-1,
    )

    print("\nTRAINING V11 STACK ...")
    stack.fit(
        X_train_encoded,
        y_train,
    )

    print("V11 stack training completed.")

    stack_validation_prob = (
        stack.predict_proba(
            X_validation_encoded
        )[:, 1]
    )

    base_validation_prob = {}

    for name in dict(estimators).keys():
        fitted_model = (
            stack.named_estimators_[name]
        )

        base_validation_prob[name] = (
            fitted_model.predict_proba(
                X_validation_encoded
            )[:, 1]
        )

    threshold_grid = np.arange(
        0.20,
        0.801,
        0.01,
    )

    search_results = []

    blend_sources = [
        ("Stacking", stack_validation_prob),
    ]

    for base_name in [
        "gb",
        "rf",
        "extra",
        "lr",
        "xgb",
    ]:
        if base_name in base_validation_prob:
            base_prob = base_validation_prob[base_name]

            for weight in [
                0.20,
                0.40,
                0.60,
                0.80,
            ]:
                blended = (
                    (1.0 - weight)
                    * stack_validation_prob
                    + weight * base_prob
                )

                label = (
                    f"Stack+{base_name}"
                    f"_base_weight={weight:.2f}"
                )

                blend_sources.append(
                    (label, blended)
                )

    print(
        "\nSEARCHING DETERMINISTIC BLENDS "
        "AND THRESHOLDS ..."
    )

    for label, probability in blend_sources:

        for threshold in threshold_grid:

            metrics = metrics_from_probability(
                y_validation,
                probability,
                float(threshold),
            )

            score = objective_score(metrics)

            search_results.append(
                {
                    "Approach": label,
                    "Threshold": float(
                        threshold
                    ),
                    **metrics,
                    "Objective_1_AllAbove": score[0],
                    "Objective_2_MinRatio": score[1],
                    "Objective_3_MeanRatio": score[2],
                }
            )

    search_df = pd.DataFrame(
        search_results
    )

    search_df = search_df.sort_values(
        by=[
            "Objective_1_AllAbove",
            "Objective_2_MinRatio",
            "Objective_3_MeanRatio",
            "F1-score",
            "ROC-AUC",
        ],
        ascending=False,
    ).reset_index(drop=True)

    best = search_df.iloc[0]

    selected_approach = best["Approach"]
    selected_threshold = float(
        best["Threshold"]
    )

    print("\nVALIDATION WINNER")
    print("-" * 90)
    print(
        f"Approach:  {selected_approach}"
    )
    print(
        f"Threshold: {selected_threshold:.2f}"
    )

    for metric in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
    ]:
        print(
            f"{metric:<12}: "
            f"{best[metric]:.4f}"
        )

    # Refit preprocessing on development data.
    X_development = pd.concat(
        [
            X_train,
            X_validation,
        ],
        axis=0,
    )

    y_development = pd.concat(
        [
            y_train,
            y_validation,
        ],
        axis=0,
    )

    final_preprocessor = make_preprocessor(
        X_development
    )

    X_development_encoded = (
        final_preprocessor.fit_transform(
            X_development
        )
    )

    X_test_encoded = (
        final_preprocessor.transform(
            X_test
        )
    )

    final_estimators = build_estimators(
        y_development
    )

    final_stack = StackingClassifier(
        estimators=final_estimators,
        final_estimator=LogisticRegression(
            C=0.25,
            max_iter=4000,
            class_weight="balanced",
            random_state=SEED,
        ),
        cv=4,
        stack_method="predict_proba",
        passthrough=True,
        n_jobs=-1,
    )

    print(
        "\nREFITTING V11 ON TRAIN + VALIDATION ..."
    )

    final_stack.fit(
        X_development_encoded,
        y_development,
    )

    final_stack_prob = (
        final_stack.predict_proba(
            X_test_encoded
        )[:, 1]
    )

    final_base_prob = {}

    for name in dict(final_estimators).keys():
        fitted_model = (
            final_stack.named_estimators_[name]
        )

        final_base_prob[name] = (
            fitted_model.predict_proba(
                X_test_encoded
            )[:, 1]
        )

    if selected_approach == "Stacking":
        final_probability = (
            final_stack_prob
        )
    else:
        parts = selected_approach.split("_base_weight=")
        base_name = (
            parts[0]
            .replace("Stack+", "")
        )

        weight = float(parts[1])

        final_probability = (
            (1.0 - weight)
            * final_stack_prob
            + weight
            * final_base_prob[base_name]
        )

    final_metrics = metrics_from_probability(
        y_test,
        final_probability,
        selected_threshold,
    )

    print_metrics(
        "FINAL V11 TEST RESULT",
        final_metrics,
    )

    dominance = sum(
        final_metrics[m]
        > REFERENCE[m]
        for m in REFERENCE
    )

    print(
        f"\nDOMINANCE COUNT: "
        f"{dominance}/5"
    )

    beats_all = dominance == 5

    print(
        "BEATS ALL FIVE ON ALL FIVE METRICS: "
        + ("YES" if beats_all else "NO")
    )

    # Compare against V9 without using it for training.
    v9_metrics = {
        "Accuracy": 0.7864,
        "Precision": 0.5855,
        "Recall": 0.6684,
        "F1-score": 0.6242,
        "ROC-AUC": 0.8441,
    }

    v9_dominance = sum(
        final_metrics[m] > v9_metrics[m]
        for m in v9_metrics
    )

    print(
        f"\nV11 vs V9 dominance: "
        f"{v9_dominance}/5"
    )

    comparison = pd.DataFrame(
        [
            {
                "Model": "SH-DAS V11",
                "Type": "Proposed",
                **final_metrics,
                "Threshold": selected_threshold,
                "Dominance_vs_existing": dominance,
            },
            {
                "Model": "SH-DAS V9",
                "Type": "Previous proposed candidate",
                **v9_metrics,
                "Threshold": 0.48,
                "Dominance_vs_existing": 2,
            },
            {
                "Model": "Existing five-model reference",
                "Type": "Reference maximum",
                **REFERENCE,
                "Threshold": np.nan,
                "Dominance_vs_existing": np.nan,
            },
        ]
    )

    comparison_path = (
        REPORT_DIR / "v11_final_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    search_path = (
        REPORT_DIR / "v11_validation_search.csv"
    )

    search_df.to_csv(
        search_path,
        index=False,
    )

    summary = {
        "version": "V11",
        "selected_approach": selected_approach,
        "selected_threshold": selected_threshold,
        "engineered_features": engineered,
        "test_metrics": final_metrics,
        "reference_metrics": REFERENCE,
        "v9_metrics": v9_metrics,
        "dominance_count": int(dominance),
        "beats_all_five": bool(beats_all),
        "v11_vs_v9_dominance": int(
            v9_dominance
        ),
        "xgboost_available": bool(
            HAS_XGB
        ),
        "train_size": int(
            len(X_train)
        ),
        "validation_size": int(
            len(X_validation)
        ),
        "test_size": int(
            len(X_test)
        ),
        "runtime_seconds": float(
            time.time() - start
        ),
    }

    summary_path = (
        REPORT_DIR / "v11_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2
        ),
        encoding="utf-8",
    )

    report_lines = [
        "SH-DAS V11 FINAL REPORT",
        "=" * 80,
        "",
        f"Selected approach: {selected_approach}",
        f"Threshold: {selected_threshold:.2f}",
        "",
        "FINAL TEST METRICS",
        "-" * 80,
    ]

    for metric in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
    ]:
        report_lines.append(
            f"{metric}: "
            f"{final_metrics[metric]:.6f} "
            f"(reference: "
            f"{REFERENCE[metric]:.6f})"
        )

    report_lines.extend(
        [
            "",
            f"Dominance: {dominance}/5",
            (
                "Beats all five: "
                + (
                    "YES"
                    if beats_all
                    else "NO"
                )
            ),
            "",
            f"V11 vs V9 dominance: "
            f"{v9_dominance}/5",
            "",
            "ENGINEERED FEATURES",
            "-" * 80,
        ]
    )

    report_lines.extend(
        f"- {item}"
        for item in engineered
    )

    report_lines.extend(
        [
            "",
            f"Runtime: "
            f"{time.time() - start:.2f} seconds",
        ]
    )

    report_path = (
        REPORT_DIR / "v11_report.txt"
    )

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    bundle_path = (
        REPORT_DIR
        / "sh_das_v11_model_bundle.joblib"
    )

    joblib.dump(
        {
            "preprocessor": final_preprocessor,
            "stacking_model": final_stack,
            "selected_approach": selected_approach,
            "threshold": selected_threshold,
            "engineered_features": engineered,
            "seed": SEED,
        },
        bundle_path,
    )

    print("\nSAVED ARTIFACTS")
    print("-" * 90)
    print(comparison_path)
    print(search_path)
    print(summary_path)
    print(report_path)
    print(bundle_path)


if __name__ == "__main__":
    main()
