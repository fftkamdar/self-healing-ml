from pathlib import Path
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SEED = 42
ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = ROOT / "reports" / "proposed_model_v12"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE = {
    "Accuracy": 0.8013,
    "Precision": 0.6679,
    "Recall": 0.7968,
    "F1-score": 0.6179,
    "ROC-AUC": 0.8430,
}

V9_REFERENCE = {
    "Accuracy": 0.7864,
    "Precision": 0.5855,
    "Recall": 0.6684,
    "F1-score": 0.6242,
    "ROC-AUC": 0.8441,
}


def find_dataset():
    candidates = [
        ROOT / "data" / "raw" / "Telco-Customer-Churn.csv",
        ROOT / "data" / "Telco-Customer-Churn.csv",
        ROOT / "data" / "Telco-Customer-Churn.csv",
        ROOT / "Telco-Customer-Churn.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    for path in ROOT.rglob("*.csv"):
        name = path.name.lower()
        if "telco" in name or "churn" in name:
            return path

    raise FileNotFoundError(
        "Telco Customer Churn CSV could not be found."
    )


def engineer_features(df):
    df = df.copy()

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce",
        )

    if "tenure" in df.columns and "TotalCharges" in df.columns:
        df["AverageMonthlySpend"] = (
            df["TotalCharges"]
            / df["tenure"].replace(0, np.nan)
        )

    services = [
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
        c for c in services if c in df.columns
    ]

    if existing_services:
        df["ServiceCount"] = 0

        for col in existing_services:
            df["ServiceCount"] += (
                ~df[col].astype(str).isin(
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

    if "Partner" in df.columns and "Dependents" in df.columns:
        df["FamilySupportCount"] = (
            df["Partner"].astype(str).eq("Yes").astype(int)
            + df["Dependents"].astype(str).eq("Yes").astype(int)
        )

    if "tenure" in df.columns and "MonthlyCharges" in df.columns:
        df["TenureChargeInteraction"] = (
            df["tenure"] * df["MonthlyCharges"]
        )

    return df


def prepare_data(df):
    if "Churn" not in df.columns:
        raise ValueError("Churn column was not found.")

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
    numeric_columns = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_columns = [
        c for c in X.columns
        if c not in numeric_columns
    ]

    numeric_pipe = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            )
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
                numeric_columns,
            ),
            (
                "cat",
                categorical_pipe,
                categorical_columns,
            ),
        ],
        remainder="drop",
    )


def metric_dict(y_true, probability, threshold):
    prediction = (
        probability >= threshold
    ).astype(int)

    return {
        "Accuracy": float(
            accuracy_score(y_true, prediction)
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


def normalized_objective(metrics):
    names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
    ]

    ratios = [
        metrics[name] / REFERENCE[name]
        for name in names
    ]

    return {
        "all_above": int(
            all(ratio > 1.0 for ratio in ratios)
        ),
        "min_ratio": float(min(ratios)),
        "mean_ratio": float(np.mean(ratios)),
        "f1": float(metrics["F1-score"]),
        "auc": float(metrics["ROC-AUC"]),
    }


def build_configs():
    configs = []

    # 1 — Conservative depth 2
    configs.append(
        (
            "XGBoost_Conservative",
            {
                "n_estimators": 500,
                "max_depth": 2,
                "learning_rate": 0.035,
                "min_child_weight": 5,
                "subsample": 0.90,
                "colsample_bytree": 0.90,
                "gamma": 0.05,
                "reg_lambda": 5.0,
                "reg_alpha": 0.10,
            },
        )
    )

    # 2 — Balanced depth 3
    configs.append(
        (
            "XGBoost_Balanced",
            {
                "n_estimators": 450,
                "max_depth": 3,
                "learning_rate": 0.035,
                "min_child_weight": 5,
                "subsample": 0.90,
                "colsample_bytree": 0.85,
                "gamma": 0.05,
                "reg_lambda": 5.0,
                "reg_alpha": 0.10,
            },
        )
    )

    # 3 — More regularized
    configs.append(
        (
            "XGBoost_Regularized",
            {
                "n_estimators": 550,
                "max_depth": 3,
                "learning_rate": 0.025,
                "min_child_weight": 8,
                "subsample": 0.85,
                "colsample_bytree": 0.80,
                "gamma": 0.10,
                "reg_lambda": 8.0,
                "reg_alpha": 0.20,
            },
        )
    )

    # 4 — Slightly more expressive
    configs.append(
        (
            "XGBoost_Expressive",
            {
                "n_estimators": 400,
                "max_depth": 4,
                "learning_rate": 0.030,
                "min_child_weight": 8,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "gamma": 0.10,
                "reg_lambda": 8.0,
                "reg_alpha": 0.15,
            },
        )
    )

    # 5 — Stronger minority emphasis
    configs.append(
        (
            "XGBoost_RecallFocused",
            {
                "n_estimators": 450,
                "max_depth": 3,
                "learning_rate": 0.030,
                "min_child_weight": 5,
                "subsample": 0.90,
                "colsample_bytree": 0.85,
                "gamma": 0.05,
                "reg_lambda": 6.0,
                "reg_alpha": 0.10,
            },
        )
    )

    return configs


def build_model(params, scale_pos_weight):
    return XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=SEED,
        n_jobs=4,
    )


def choose_threshold(y_true, probability):
    threshold_grid = np.arange(
        0.20,
        0.801,
        0.01,
    )

    records = []

    for threshold in threshold_grid:
        metrics = metric_dict(
            y_true,
            probability,
            float(threshold),
        )

        obj = normalized_objective(metrics)

        records.append(
            {
                "Threshold": float(threshold),
                **metrics,
                "AllAbove": obj["all_above"],
                "MinRatio": obj["min_ratio"],
                "MeanRatio": obj["mean_ratio"],
            }
        )

    result = pd.DataFrame(records)

    result = result.sort_values(
        by=[
            "AllAbove",
            "MinRatio",
            "MeanRatio",
            "F1-score",
            "ROC-AUC",
        ],
        ascending=False,
    ).reset_index(drop=True)

    return result.iloc[0], result


def main():
    start = time.time()

    print("=" * 90)
    print("SH-DAS V12 - XGBOOST GENERALIZATION SEARCH")
    print("=" * 90)

    dataset = find_dataset()

    print(f"\nDataset: {dataset}")

    df = pd.read_csv(dataset)

    # Engineered dataset
    engineered_df = engineer_features(df)

    X, y = prepare_data(engineered_df)

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

    print(f"\nTrain:      {len(X_train):,}")
    print(f"Validation: {len(X_validation):,}")
    print(f"Development:{len(X_development):,}")
    print(f"Test:       {len(X_test):,}")

    engineered_features = [
        c
        for c in [
            "AverageMonthlySpend",
            "ServiceCount",
            "IsMonthToMonth",
            "FamilySupportCount",
            "TenureChargeInteraction",
        ]
        if c in X.columns
    ]

    print("\nEngineered features:")
    for feature in engineered_features:
        print(f"  + {feature}")

    positive = int(y_development.sum())
    negative = int(
        len(y_development) - positive
    )

    scale_pos_weight = (
        negative / positive
        if positive > 0
        else 1.0
    )

    print(
        f"\nScale positive weight: "
        f"{scale_pos_weight:.4f}"
    )

    preprocessor = make_preprocessor(
        X_development
    )

    X_dev_encoded = preprocessor.fit_transform(
        X_development
    )

    X_test_encoded = preprocessor.transform(
        X_test
    )

    configs = build_configs()

    folds = StratifiedKFold(
        n_splits=4,
        shuffle=True,
        random_state=SEED,
    )

    search_records = []
    threshold_records = {}

    print("\n4-FOLD OOF CONFIGURATION SEARCH")
    print("-" * 90)

    for config_name, params in configs:
        print(
            f"\nRunning {config_name} ..."
        )

        oof_probability = np.zeros(
            len(X_dev_encoded),
            dtype=float,
        )

        fold_auc = []

        for fold_number, (
            train_indices,
            valid_indices,
        ) in enumerate(
            folds.split(
                X_dev_encoded,
                y_development,
            ),
            start=1,
        ):
            model = build_model(
                params,
                scale_pos_weight,
            )

            model.fit(
                X_dev_encoded[train_indices],
                y_development.iloc[
                    train_indices
                ],
            )

            fold_probability = (
                model.predict_proba(
                    X_dev_encoded[valid_indices]
                )[:, 1]
            )

            oof_probability[
                valid_indices
            ] = fold_probability

            fold_auc.append(
                roc_auc_score(
                    y_development.iloc[
                        valid_indices
                    ],
                    fold_probability,
                )
            )

            print(
                f"  Fold {fold_number}: "
                f"AUC={fold_auc[-1]:.4f}"
            )

        best_row, threshold_df = (
            choose_threshold(
                y_development,
                oof_probability,
            )
        )

        metrics = {
            metric: float(best_row[metric])
            for metric in [
                "Accuracy",
                "Precision",
                "Recall",
                "F1-score",
                "ROC-AUC",
            ]
        }

        obj = normalized_objective(metrics)

        record = {
            "Model": config_name,
            "Threshold": float(
                best_row["Threshold"]
            ),
            **metrics,
            "OOF_AUC_Mean": float(
                np.mean(fold_auc)
            ),
            "OOF_AUC_SD": float(
                np.std(fold_auc)
            ),
            "AllAbove": obj["all_above"],
            "MinRatio": obj["min_ratio"],
            "MeanRatio": obj["mean_ratio"],
        }

        search_records.append(record)

        threshold_records[
            config_name
        ] = threshold_df

        print(
            f"  OOF Accuracy={metrics['Accuracy']:.4f}"
        )
        print(
            f"  OOF Precision={metrics['Precision']:.4f}"
        )
        print(
            f"  OOF Recall={metrics['Recall']:.4f}"
        )
        print(
            f"  OOF F1={metrics['F1-score']:.4f}"
        )
        print(
            f"  OOF AUC={metrics['ROC-AUC']:.4f}"
        )
        print(
            f"  Threshold={best_row['Threshold']:.2f}"
        )

    search_df = pd.DataFrame(
        search_records
    )

    search_df = search_df.sort_values(
        by=[
            "AllAbove",
            "MinRatio",
            "MeanRatio",
            "F1-score",
            "ROC-AUC",
        ],
        ascending=False,
    ).reset_index(drop=True)

    selected = search_df.iloc[0]

    selected_name = selected["Model"]
    selected_threshold = float(
        selected["Threshold"]
    )

    print("\nOOF SELECTED CONFIGURATION")
    print("-" * 90)
    print(f"Model:     {selected_name}")
    print(
        f"Threshold: {selected_threshold:.2f}"
    )

    selected_params = None

    for name, params in configs:
        if name == selected_name:
            selected_params = params
            break

    print(
        "\nFITTING FINAL XGBOOST ON "
        "FULL DEVELOPMENT DATA ..."
    )

    final_model = build_model(
        selected_params,
        scale_pos_weight,
    )

    final_model.fit(
        X_dev_encoded,
        y_development,
    )

    test_probability = (
        final_model.predict_proba(
            X_test_encoded
        )[:, 1]
    )

    test_metrics = metric_dict(
        y_test,
        test_probability,
        selected_threshold,
    )

    print("\n" + "=" * 90)
    print("FINAL V12 RESULT")
    print("=" * 90)

    for metric in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
    ]:
        actual = test_metrics[metric]
        reference = REFERENCE[metric]
        symbol = (
            "PASS"
            if actual > reference
            else "MISS"
        )

        print(
            f"{metric:<12}: "
            f"{actual:.4f} "
            f"(existing max: {reference:.4f}) "
            f"{symbol}"
        )

    dominance = sum(
        test_metrics[metric]
        > REFERENCE[metric]
        for metric in REFERENCE
    )

    v9_dominance = sum(
        test_metrics[metric]
        > V9_REFERENCE[metric]
        for metric in V9_REFERENCE
    )

    print(
        f"\nDOMINANCE COUNT: "
        f"{dominance}/5"
    )

    print(
        "BEATS ALL FIVE ON ALL FIVE METRICS: "
        + (
            "YES"
            if dominance == 5
            else "NO"
        )
    )

    print(
        f"V12 vs V9 dominance: "
        f"{v9_dominance}/5"
    )

    comparison = pd.DataFrame(
        [
            {
                "Model": "SH-DAS V12",
                "Type": "XGBoost Proposed",
                **test_metrics,
                "Threshold": selected_threshold,
                "Dominance_vs_existing": dominance,
            },
            {
                "Model": "SH-DAS V9",
                "Type": "Previous proposed candidate",
                **V9_REFERENCE,
                "Threshold": 0.48,
                "Dominance_vs_existing": 2,
            },
            {
                "Model": "Existing Best Reference",
                "Type": "Five-model reference",
                **REFERENCE,
                "Threshold": np.nan,
                "Dominance_vs_existing": np.nan,
            },
        ]
    )

    comparison_path = (
        REPORT_DIR / "v12_final_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    search_path = (
        REPORT_DIR / "v12_oof_configuration_search.csv"
    )

    search_df.to_csv(
        search_path,
        index=False,
    )

    summary = {
        "version": "V12",
        "selected_model": selected_name,
        "selected_threshold": selected_threshold,
        "selected_parameters": selected_params,
        "engineered_features": engineered_features,
        "oof_selection": {
            "Accuracy": float(
                selected["Accuracy"]
            ),
            "Precision": float(
                selected["Precision"]
            ),
            "Recall": float(
                selected["Recall"]
            ),
            "F1-score": float(
                selected["F1-score"]
            ),
            "ROC-AUC": float(
                selected["ROC-AUC"]
            ),
        },
        "test_metrics": test_metrics,
        "reference_metrics": REFERENCE,
        "v9_metrics": V9_REFERENCE,
        "dominance_count": int(
            dominance
        ),
        "beats_all_five": bool(
            dominance == 5
        ),
        "v12_vs_v9_dominance": int(
            v9_dominance
        ),
        "runtime_seconds": float(
            time.time() - start
        ),
    }

    summary_path = (
        REPORT_DIR / "v12_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "SH-DAS V12 FINAL REPORT",
        "=" * 80,
        "",
        f"Selected model: {selected_name}",
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
            f"{test_metrics[metric]:.6f} "
            f"(existing reference: "
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
                    if dominance == 5
                    else "NO"
                )
            ),
            "",
            f"V12 vs V9 dominance: "
            f"{v9_dominance}/5",
            "",
            "ENGINEERED FEATURES",
            "-" * 80,
        ]
    )

    report_lines.extend(
        f"- {feature}"
        for feature in engineered_features
    )

    report_path = (
        REPORT_DIR / "v12_report.txt"
    )

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    bundle_path = (
        REPORT_DIR
        / "sh_das_v12_model_bundle.joblib"
    )

    joblib.dump(
        {
            "preprocessor": preprocessor,
            "model": final_model,
            "threshold": selected_threshold,
            "selected_model": selected_name,
            "engineered_features": engineered_features,
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
