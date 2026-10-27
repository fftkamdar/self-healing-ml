from pathlib import Path
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings("ignore")

SEED = 42

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "proposed_model_v15"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# FROZEN CANONICAL BENCHMARK
REFERENCE = {
    "Accuracy": 0.791341,
    "Precision": 0.639860,
    "Recall": 0.810160,
    "F1-score": 0.624742,
    "ROC-AUC": 0.835577,
}

# CURRENT BEST PROPOSED RESULT
V9 = {
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
        "Telco Customer Churn CSV not found."
    )


def prepare_dataset(df):
    df = df.copy()

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce",
        )

    # Carefully chosen numerical interactions
    if "tenure" in df.columns and "MonthlyCharges" in df.columns:
        df["TenureChargeInteraction"] = (
            df["tenure"] * df["MonthlyCharges"]
        )

    if "tenure" in df.columns and "TotalCharges" in df.columns:
        df["AverageMonthlySpend"] = (
            df["TotalCharges"]
            / df["tenure"].replace(0, np.nan)
        )

    # Contract indicator
    if "Contract" in df.columns:
        df["IsMonthToMonth"] = (
            df["Contract"]
            .astype(str)
            .eq("Month-to-month")
            .astype(int)
        )

    # Service count
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
        col
        for col in services
        if col in df.columns
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

    # Family support
    if (
        "Partner" in df.columns
        and "Dependents" in df.columns
    ):
        df["FamilySupportCount"] = (
            df["Partner"]
            .astype(str)
            .eq("Yes")
            .astype(int)
            +
            df["Dependents"]
            .astype(str)
            .eq("Yes")
            .astype(int)
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
            "Unexpected Churn values."
        )

    X = df.drop(columns=["Churn"])

    return X, y.astype(int)


def convert_categoricals(X):
    X = X.copy()

    categorical = []

    for col in X.columns:
        # Treat every non-numeric, non-boolean column as categorical.
        # This also handles pandas string dtypes.
        if (
            not pd.api.types.is_numeric_dtype(X[col])
            and not pd.api.types.is_bool_dtype(X[col])
        ):
            X[col] = (
                X[col]
                .fillna("Missing")
                .astype("category")
            )
            categorical.append(col)

    return X, categorical


def make_model(params):
    return LGBMClassifier(
        objective="binary",
        random_state=SEED,
        n_jobs=4,
        verbosity=-1,
        **params,
    )


def calculate_metrics(
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


def objective(metrics):
    ratios = [
        metrics[m] / REFERENCE[m]
        for m in REFERENCE
    ]

    margins = [
        metrics[m] - REFERENCE[m]
        for m in REFERENCE
    ]

    return (
        int(
            all(
                r > 1.0
                for r in ratios
            )
        ),
        float(min(ratios)),
        float(np.mean(ratios)),
        float(np.mean(margins)),
        float(metrics["F1-score"]),
        float(metrics["ROC-AUC"]),
    )


def threshold_search(y_true, probability):
    records = []

    for threshold in np.arange(
        0.20,
        0.801,
        0.005,
    ):
        metrics = calculate_metrics(
            y_true,
            probability,
            float(threshold),
        )

        score = objective(metrics)

        records.append(
            {
                "Threshold": float(threshold),
                **metrics,
                "AllAbove": score[0],
                "MinRatio": score[1],
                "MeanRatio": score[2],
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


def build_configs():
    return [
        (
            "LGBM_Conservative",
            {
                "n_estimators": 600,
                "learning_rate": 0.025,
                "num_leaves": 12,
                "max_depth": 4,
                "min_child_samples": 30,
                "subsample": 0.90,
                "colsample_bytree": 0.90,
                "reg_alpha": 0.10,
                "reg_lambda": 5.0,
                "class_weight": None,
            },
        ),
        (
            "LGBM_Balanced",
            {
                "n_estimators": 550,
                "learning_rate": 0.030,
                "num_leaves": 15,
                "max_depth": 5,
                "min_child_samples": 30,
                "subsample": 0.90,
                "colsample_bytree": 0.90,
                "reg_alpha": 0.10,
                "reg_lambda": 5.0,
                "class_weight": None,
            },
        ),
        (
            "LGBM_Regularized",
            {
                "n_estimators": 700,
                "learning_rate": 0.020,
                "num_leaves": 10,
                "max_depth": 4,
                "min_child_samples": 35,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_alpha": 0.20,
                "reg_lambda": 8.0,
                "class_weight": None,
            },
        ),
        (
            "LGBM_RecallBalanced",
            {
                "n_estimators": 600,
                "learning_rate": 0.025,
                "num_leaves": 12,
                "max_depth": 4,
                "min_child_samples": 25,
                "subsample": 0.90,
                "colsample_bytree": 0.90,
                "reg_alpha": 0.10,
                "reg_lambda": 6.0,
                "class_weight": {
                    0: 1.0,
                    1: 1.20,
                },
            },
        ),
        (
            "LGBM_PrecisionBalanced",
            {
                "n_estimators": 500,
                "learning_rate": 0.030,
                "num_leaves": 12,
                "max_depth": 4,
                "min_child_samples": 35,
                "subsample": 0.90,
                "colsample_bytree": 0.90,
                "reg_alpha": 0.15,
                "reg_lambda": 7.0,
                "class_weight": {
                    0: 1.0,
                    1: 0.90,
                },
            },
        ),
        (
            "LGBM_AccuracyFocused",
            {
                "n_estimators": 500,
                "learning_rate": 0.030,
                "num_leaves": 10,
                "max_depth": 4,
                "min_child_samples": 40,
                "subsample": 0.95,
                "colsample_bytree": 0.95,
                "reg_alpha": 0.10,
                "reg_lambda": 6.0,
                "class_weight": {
                    0: 1.0,
                    1: 0.95,
                },
            },
        ),
    ]


def main():
    start = time.time()

    print("=" * 100)
    print("SH-DAS V15 — TARGETED LIGHTGBM CHAMPION SEARCH")
    print("=" * 100)

    dataset = find_dataset()

    print(
        f"\nDataset: {dataset}"
    )

    df = pd.read_csv(dataset)

    X, y = prepare_dataset(df)

    X, categorical_columns = (
        convert_categoricals(X)
    )

    # EXACT SAME SPLIT AS CANONICAL BENCHMARK
    X_train, X_temp, y_train, y_temp = (
        train_test_split(
            X,
            y,
            test_size=0.40,
            stratify=y,
            random_state=SEED,
        )
    )

    X_validation, X_test, y_validation, y_test = (
        train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            stratify=y_temp,
            random_state=SEED,
        )
    )

    # Development set for OOF selection
    X_development = pd.concat(
        [
            X_train,
            X_validation,
        ],
        axis=0,
    ).reset_index(drop=True)

    y_development = pd.concat(
        [
            y_train,
            y_validation,
        ],
        axis=0,
    ).reset_index(drop=True)

    X_test = X_test.reset_index(
        drop=True
    )

    y_test = y_test.reset_index(
        drop=True
    )

    print(
        f"\nTrain:       {len(X_train):,}"
    )
    print(
        f"Validation:  {len(X_validation):,}"
    )
    print(
        f"Development: {len(X_development):,}"
    )
    print(
        f"Test:        {len(X_test):,}"
    )

    print("\nCategorical features:")
    for col in categorical_columns:
        print(f"  + {col}")

    engineered_features = [
        c
        for c in [
            "AverageMonthlySpend",
            "TenureChargeInteraction",
            "IsMonthToMonth",
            "ServiceCount",
            "FamilySupportCount",
        ]
        if c in X.columns
    ]

    print("\nEngineered features:")
    for col in engineered_features:
        print(f"  + {col}")

    configs = build_configs()

    folds = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    config_results = []

    oof_predictions = {}

    print("\n5-FOLD OOF SEARCH")
    print("-" * 100)

    for config_name, params in configs:

        print(
            f"\nRunning {config_name} ..."
        )

        oof_probability = np.zeros(
            len(X_development),
            dtype=float,
        )

        fold_auc = []

        for fold_no, (
            train_idx,
            valid_idx,
        ) in enumerate(
            folds.split(
                X_development,
                y_development,
            ),
            start=1,
        ):

            model = make_model(
                params
            )

            X_fold_train = (
                X_development.iloc[
                    train_idx
                ]
            )

            X_fold_valid = (
                X_development.iloc[
                    valid_idx
                ]
            )

            y_fold_train = (
                y_development.iloc[
                    train_idx
                ]
            )

            y_fold_valid = (
                y_development.iloc[
                    valid_idx
                ]
            )

            model.fit(
                X_fold_train,
                y_fold_train,
                categorical_feature=categorical_columns,
            )

            probability = (
                model.predict_proba(
                    X_fold_valid
                )[:, 1]
            )

            oof_probability[
                valid_idx
            ] = probability

            auc = roc_auc_score(
                y_fold_valid,
                probability,
            )

            fold_auc.append(
                auc
            )

            print(
                f"  Fold {fold_no}: "
                f"AUC={auc:.4f}"
            )

        oof_predictions[
            config_name
        ] = oof_probability

        best_row, _ = threshold_search(
            y_development,
            oof_probability,
        )

        metrics = {
            metric: float(
                best_row[metric]
            )
            for metric in REFERENCE
        }

        score = objective(
            metrics
        )

        record = {
            "Model": config_name,
            "Threshold": float(
                best_row["Threshold"]
            ),
            **metrics,
            "MeanFoldAUC": float(
                np.mean(fold_auc)
            ),
            "StdFoldAUC": float(
                np.std(fold_auc)
            ),
            "AllAbove": score[0],
            "MinRatio": score[1],
            "MeanRatio": score[2],
        }

        config_results.append(
            record
        )

        print(
            f"  OOF Accuracy="
            f"{metrics['Accuracy']:.4f} "
            f"Precision="
            f"{metrics['Precision']:.4f} "
            f"Recall="
            f"{metrics['Recall']:.4f} "
            f"F1="
            f"{metrics['F1-score']:.4f} "
            f"AUC="
            f"{metrics['ROC-AUC']:.4f} "
            f"Threshold="
            f"{best_row['Threshold']:.3f}"
        )

    results_df = pd.DataFrame(
        config_results
    )

    results_df = results_df.sort_values(
        by=[
            "AllAbove",
            "MinRatio",
            "MeanRatio",
            "F1-score",
            "ROC-AUC",
        ],
        ascending=False,
    ).reset_index(drop=True)

    selected = results_df.iloc[0]

    selected_name = selected["Model"]

    selected_threshold = float(
        selected["Threshold"]
    )

    selected_params = None

    for name, params in configs:
        if name == selected_name:
            selected_params = params
            break

    print("\n" + "=" * 100)
    print("OOF SELECTED CONFIGURATION")
    print("=" * 100)

    print(
        f"Model:     {selected_name}"
    )

    print(
        f"Threshold: {selected_threshold:.3f}"
    )

    for metric in REFERENCE:
        print(
            f"{metric:<12}: "
            f"{selected[metric]:.4f}"
        )

    # ---------------------------------------------------------
    # FINAL MODEL
    # ---------------------------------------------------------

    print(
        "\nFITTING FINAL LIGHTGBM "
        "ON FULL DEVELOPMENT DATA ..."
    )

    final_model = make_model(
        selected_params
    )

    final_model.fit(
        X_development,
        y_development,
        categorical_feature=categorical_columns,
    )

    test_probability = (
        final_model.predict_proba(
            X_test
        )[:, 1]
    )

    final_metrics = calculate_metrics(
        y_test,
        test_probability,
        selected_threshold,
    )

    dominance = sum(
        final_metrics[m]
        > REFERENCE[m]
        for m in REFERENCE
    )

    v9_dominance = sum(
        final_metrics[m]
        > V9[m]
        for m in V9
    )

    print("\n" + "=" * 100)
    print("FINAL V15 RESULT")
    print("=" * 100)

    print(
        f"Selected model: {selected_name}"
    )

    print(
        f"Threshold: {selected_threshold:.3f}"
    )

    for metric in REFERENCE:

        actual = final_metrics[metric]

        reference = REFERENCE[metric]

        symbol = (
            "PASS"
            if actual > reference
            else "MISS"
        )

        print(
            f"{metric:<12}: "
            f"{actual:.4f} "
            f"(canonical max: "
            f"{reference:.4f}) "
            f"{symbol}"
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
        f"V15 vs V9 dominance: "
        f"{v9_dominance}/5"
    )

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    comparison = pd.DataFrame(
        [
            {
                "Model": "SH-DAS V15",
                "Type": "LightGBM Proposed",
                **final_metrics,
                "Threshold": selected_threshold,
                "CanonicalDominance": dominance,
                "V15_vs_V9_Dominance": v9_dominance,
            },
            {
                "Model": "SH-DAS V9",
                "Type": "Current proposed benchmark",
                **V9,
                "Threshold": 0.48,
                "CanonicalDominance": np.nan,
                "V15_vs_V9_Dominance": np.nan,
            },
            {
                "Model": "Canonical Maximum",
                "Type": "Frozen five-model benchmark",
                **REFERENCE,
                "Threshold": np.nan,
                "CanonicalDominance": np.nan,
                "V15_vs_V9_Dominance": np.nan,
            },
        ]
    )

    comparison_path = (
        REPORT_DIR
        / "v15_final_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    search_path = (
        REPORT_DIR
        / "v15_oof_search.csv"
    )

    results_df.to_csv(
        search_path,
        index=False,
    )

    summary = {
        "version": "V15",
        "selected_model": selected_name,
        "selected_threshold": selected_threshold,
        "selected_parameters": selected_params,
        "categorical_features": categorical_columns,
        "engineered_features": engineered_features,
        "test_metrics": final_metrics,
        "canonical_reference": REFERENCE,
        "v9_reference": V9,
        "dominance_count": int(
            dominance
        ),
        "beats_all_five": bool(
            dominance == 5
        ),
        "v15_vs_v9_dominance": int(
            v9_dominance
        ),
        "train_size": int(
            len(X_train)
        ),
        "validation_size": int(
            len(X_validation)
        ),
        "development_size": int(
            len(X_development)
        ),
        "test_size": int(
            len(X_test)
        ),
        "runtime_seconds": float(
            time.time() - start
        ),
    }

    summary_path = (
        REPORT_DIR
        / "v15_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "SH-DAS V15 FINAL REPORT",
        "=" * 80,
        "",
        f"Selected model: {selected_name}",
        f"Threshold: {selected_threshold:.3f}",
        "",
        "FINAL TEST METRICS",
        "-" * 80,
    ]

    for metric in REFERENCE:
        report_lines.append(
            f"{metric}: "
            f"{final_metrics[metric]:.6f} "
            f"(canonical maximum: "
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
            f"V15 vs V9 dominance: "
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

    report_lines.extend(
        [
            "",
            "NATIVE CATEGORICAL FEATURES",
            "-" * 80,
        ]
    )

    report_lines.extend(
        f"- {feature}"
        for feature in categorical_columns
    )

    report_lines.extend(
        [
            "",
            f"Runtime: "
            f"{time.time() - start:.2f} seconds",
        ]
    )

    report_path = (
        REPORT_DIR
        / "v15_report.txt"
    )

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    bundle_path = (
        REPORT_DIR
        / "sh_das_v15_model_bundle.joblib"
    )

    joblib.dump(
        {
            "model": final_model,
            "threshold": selected_threshold,
            "selected_model": selected_name,
            "categorical_features": categorical_columns,
            "engineered_features": engineered_features,
            "seed": SEED,
        },
        bundle_path,
    )

    print("\nSAVED")
    print("-" * 100)
    print(comparison_path)
    print(search_path)
    print(summary_path)
    print(report_path)
    print(bundle_path)


if __name__ == "__main__":
    main()

