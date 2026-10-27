from pathlib import Path
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

SEED = 42

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "hybrid_final_candidate"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# FROZEN CANONICAL BENCHMARK
REFERENCE = {
    "Accuracy": 0.791341,
    "Precision": 0.639860,
    "Recall": 0.810160,
    "F1-score": 0.624742,
    "ROC-AUC": 0.835577,
}

# CURRENT PROPOSED BENCHMARK
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
            "Unexpected Churn values found."
        )

    X = df.drop(columns=["Churn"])

    return X, y.astype(int)


def make_preprocessor(X):
    numeric = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical = [
        c
        for c in X.columns
        if c not in numeric
    ]

    numeric_pipe = Pipeline(
        [
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

    categorical_pipe = Pipeline(
        [
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
        [
            (
                "num",
                numeric_pipe,
                numeric,
            ),
            (
                "cat",
                categorical_pipe,
                categorical,
            ),
        ]
    )


def make_lr(config):
    return LogisticRegression(
        C=config["C"],
        class_weight=config["class_weight"],
        max_iter=4000,
        random_state=SEED,
    )


def make_gb(config):
    return GradientBoostingClassifier(
        n_estimators=config["n_estimators"],
        learning_rate=config["learning_rate"],
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        subsample=config["subsample"],
        random_state=SEED,
    )


def build_pipeline(model):
    return Pipeline(
        [
            (
                "preprocessor",
                make_preprocessor(current_X),
            ),
            (
                "model",
                model,
            ),
        ]
    )


def calculate_metrics(y_true, probability, threshold):
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
                ratio > 1.0
                for ratio in ratios
            )
        ),
        float(min(ratios)),
        float(np.mean(ratios)),
        float(np.mean(margins)),
        float(metrics["F1-score"]),
        float(metrics["ROC-AUC"]),
    )


def main():
    global current_X

    start = time.time()

    print("=" * 100)
    print("SH-DAS HYBRID FINAL CANDIDATE")
    print("=" * 100)

    dataset = find_dataset()

    print(
        f"\nDataset: {dataset}"
    )

    df = pd.read_csv(dataset)

    current_X, y = prepare_dataset(df)

    # EXACT CANONICAL SPLIT
    X_train, X_temp, y_train, y_temp = (
        train_test_split(
            current_X,
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

    print(
        f"\nTrain:      {len(X_train):,}"
    )
    print(
        f"Validation: {len(X_validation):,}"
    )
    print(
        f"Test:       {len(X_test):,}"
    )

    # ---------------------------------------------------------
    # Candidate configurations
    # ---------------------------------------------------------

    lr_configs = [
        (
            "LR_Balanced_C0.5",
            {
                "C": 0.5,
                "class_weight": "balanced",
            },
        ),
        (
            "LR_LightBalance",
            {
                "C": 1.0,
                "class_weight": {
                    0: 1.0,
                    1: 1.35,
                },
            },
        ),
        (
            "LR_ModerateBalance",
            {
                "C": 0.5,
                "class_weight": {
                    0: 1.0,
                    1: 1.60,
                },
            },
        ),
    ]

    gb_configs = [
        (
            "GB_Canonical",
            {
                "n_estimators": 180,
                "learning_rate": 0.05,
                "max_depth": 2,
                "min_samples_leaf": 8,
                "subsample": 1.0,
            },
        ),
        (
            "GB_Conservative",
            {
                "n_estimators": 250,
                "learning_rate": 0.03,
                "max_depth": 2,
                "min_samples_leaf": 10,
                "subsample": 0.90,
            },
        ),
        (
            "GB_Smooth",
            {
                "n_estimators": 220,
                "learning_rate": 0.035,
                "max_depth": 2,
                "min_samples_leaf": 12,
                "subsample": 0.90,
            },
        ),
    ]

    folds = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    # ---------------------------------------------------------
    # OOF evaluation on TRAIN ONLY
    # ---------------------------------------------------------

    print("\n5-FOLD OOF MODEL SEARCH")
    print("-" * 100)

    oof_predictions = {}
    model_info = []

    all_configs = []

    for name, config in lr_configs:
        all_configs.append(
            (
                name,
                "LR",
                config,
            )
        )

    for name, config in gb_configs:
        all_configs.append(
            (
                name,
                "GB",
                config,
            )
        )

    for name, kind, config in all_configs:

        print(
            f"\nRunning {name} ..."
        )

        oof = np.zeros(
            len(X_train),
            dtype=float,
        )

        fold_auc = []

        for fold_no, (
            train_idx,
            valid_idx,
        ) in enumerate(
            folds.split(
                X_train,
                y_train,
            ),
            start=1,
        ):

            X_fold_train = X_train.iloc[
                train_idx
            ]

            X_fold_valid = X_train.iloc[
                valid_idx
            ]

            y_fold_train = y_train.iloc[
                train_idx
            ]

            y_fold_valid = y_train.iloc[
                valid_idx
            ]

            if kind == "LR":
                model = make_lr(config)
            else:
                model = make_gb(config)

            pipe = Pipeline(
                [
                    (
                        "preprocessor",
                        make_preprocessor(
                            X_fold_train
                        ),
                    ),
                    (
                        "model",
                        model,
                    ),
                ]
            )

            pipe.fit(
                X_fold_train,
                y_fold_train,
            )

            fold_probability = (
                pipe.predict_proba(
                    X_fold_valid
                )[:, 1]
            )

            oof[
                valid_idx
            ] = fold_probability

            fold_auc_value = (
                roc_auc_score(
                    y_fold_valid,
                    fold_probability,
                )
            )

            fold_auc.append(
                fold_auc_value
            )

        # OOF threshold search
        best = None

        for threshold in np.arange(
            0.20,
            0.801,
            0.01,
        ):

            m = calculate_metrics(
                y_train,
                oof,
                float(threshold),
            )

            score = objective(m)

            record = {
                "Model": name,
                "Kind": kind,
                "Threshold": float(
                    threshold
                ),
                **m,
                "OOFMeanFoldAUC": float(
                    np.mean(fold_auc)
                ),
                "OOFStdFoldAUC": float(
                    np.std(fold_auc)
                ),
                "AllAbove": score[0],
                "MinRatio": score[1],
                "MeanRatio": score[2],
            }

            if best is None:
                best = record
            else:
                old_score = (
                    best["AllAbove"],
                    best["MinRatio"],
                    best["MeanRatio"],
                    best["F1-score"],
                    best["ROC-AUC"],
                )

                new_score = (
                    record["AllAbove"],
                    record["MinRatio"],
                    record["MeanRatio"],
                    record["F1-score"],
                    record["ROC-AUC"],
                )

                if new_score > old_score:
                    best = record

        oof_predictions[name] = oof
        model_info.append(best)

        print(
            f"  OOF Accuracy={best['Accuracy']:.4f} "
            f"Precision={best['Precision']:.4f} "
            f"Recall={best['Recall']:.4f} "
            f"F1={best['F1-score']:.4f} "
            f"AUC={best['ROC-AUC']:.4f} "
            f"Threshold={best['Threshold']:.2f}"
        )

    model_df = pd.DataFrame(
        model_info
    )

    # ---------------------------------------------------------
    # Select strongest LR and strongest GB by OOF objective
    # ---------------------------------------------------------

    model_df_sorted = model_df.sort_values(
        by=[
            "AllAbove",
            "MinRatio",
            "MeanRatio",
            "F1-score",
            "ROC-AUC",
        ],
        ascending=False,
    )

    best_lr_row = (
        model_df_sorted[
            model_df_sorted["Kind"] == "LR"
        ]
        .iloc[0]
    )

    best_gb_row = (
        model_df_sorted[
            model_df_sorted["Kind"] == "GB"
        ]
        .iloc[0]
    )

    lr_name = best_lr_row["Model"]
    gb_name = best_gb_row["Model"]

    print("\nSELECTED OOF COMPONENTS")
    print("-" * 100)
    print(
        f"Logistic Regression: {lr_name}"
    )
    print(
        f"Gradient Boosting:   {gb_name}"
    )

    # ---------------------------------------------------------
    # Deterministic blend search on OOF predictions
    # ---------------------------------------------------------

    lr_oof = oof_predictions[
        lr_name
    ]

    gb_oof = oof_predictions[
        gb_name
    ]

    blend_best = None

    print(
        "\nSEARCHING LR + GB BLENDS"
    )
    print("-" * 100)

    for gb_weight in np.arange(
        0.0,
        1.001,
        0.05,
    ):

        lr_weight = 1.0 - gb_weight

        blended_oof = (
            lr_weight * lr_oof
            + gb_weight * gb_oof
        )

        for threshold in np.arange(
            0.20,
            0.801,
            0.01,
        ):

            m = calculate_metrics(
                y_train,
                blended_oof,
                float(threshold),
            )

            score = objective(m)

            record = {
                "LRWeight": float(
                    lr_weight
                ),
                "GBWeight": float(
                    gb_weight
                ),
                "Threshold": float(
                    threshold
                ),
                **m,
                "AllAbove": score[0],
                "MinRatio": score[1],
                "MeanRatio": score[2],
            }

            if blend_best is None:
                blend_best = record
            else:
                old_score = (
                    blend_best["AllAbove"],
                    blend_best["MinRatio"],
                    blend_best["MeanRatio"],
                    blend_best["F1-score"],
                    blend_best["ROC-AUC"],
                )

                new_score = (
                    record["AllAbove"],
                    record["MinRatio"],
                    record["MeanRatio"],
                    record["F1-score"],
                    record["ROC-AUC"],
                )

                if new_score > old_score:
                    blend_best = record

    print("\nOOF BEST HYBRID")
    print("-" * 100)

    print(
        f"LR weight: "
        f"{blend_best['LRWeight']:.2f}"
    )

    print(
        f"GB weight: "
        f"{blend_best['GBWeight']:.2f}"
    )

    print(
        f"Threshold: "
        f"{blend_best['Threshold']:.2f}"
    )

    for metric in REFERENCE:
        print(
            f"{metric:<12}: "
            f"{blend_best[metric]:.4f}"
        )

    # ---------------------------------------------------------
    # Train the selected LR and GB on canonical TRAIN
    # and evaluate on VALIDATION to determine whether
    # the selected blend transfers.
    # ---------------------------------------------------------

    lr_config = None
    gb_config = None

    for name, config in lr_configs:
        if name == lr_name:
            lr_config = config

    for name, config in gb_configs:
        if name == gb_name:
            gb_config = config

    print(
        "\nTRAINING SELECTED COMPONENTS "
        "ON FULL TRAIN SET ..."
    )

    lr_pipe = Pipeline(
        [
            (
                "preprocessor",
                make_preprocessor(
                    X_train
                ),
            ),
            (
                "model",
                make_lr(
                    lr_config
                ),
            ),
        ]
    )

    gb_pipe = Pipeline(
        [
            (
                "preprocessor",
                make_preprocessor(
                    X_train
                ),
            ),
            (
                "model",
                make_gb(
                    gb_config
                ),
            ),
        ]
    )

    lr_pipe.fit(
        X_train,
        y_train,
    )

    gb_pipe.fit(
        X_train,
        y_train,
    )

    validation_lr = (
        lr_pipe.predict_proba(
            X_validation
        )[:, 1]
    )

    validation_gb = (
        gb_pipe.predict_proba(
            X_validation
        )[:, 1]
    )

    validation_probability = (
        blend_best["LRWeight"]
        * validation_lr
        +
        blend_best["GBWeight"]
        * validation_gb
    )

    validation_metrics = calculate_metrics(
        y_validation,
        validation_probability,
        blend_best["Threshold"],
    )

    print("\nVALIDATION TRANSFER CHECK")
    print("-" * 100)

    for metric in REFERENCE:
        print(
            f"{metric:<12}: "
            f"{validation_metrics[metric]:.4f}"
        )

    # ---------------------------------------------------------
    # FINAL REFIT ON TRAIN + VALIDATION
    # ---------------------------------------------------------

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

    print(
        "\nFINAL REFIT ON TRAIN + VALIDATION ..."
    )

    final_lr = Pipeline(
        [
            (
                "preprocessor",
                make_preprocessor(
                    X_development
                ),
            ),
            (
                "model",
                make_lr(
                    lr_config
                ),
            ),
        ]
    )

    final_gb = Pipeline(
        [
            (
                "preprocessor",
                make_preprocessor(
                    X_development
                ),
            ),
            (
                "model",
                make_gb(
                    gb_config
                ),
            ),
        ]
    )

    final_lr.fit(
        X_development,
        y_development,
    )

    final_gb.fit(
        X_development,
        y_development,
    )

    test_lr = (
        final_lr.predict_proba(
            X_test
        )[:, 1]
    )

    test_gb = (
        final_gb.predict_proba(
            X_test
        )[:, 1]
    )

    final_probability = (
        blend_best["LRWeight"]
        * test_lr
        +
        blend_best["GBWeight"]
        * test_gb
    )

    final_threshold = float(
        blend_best["Threshold"]
    )

    final_metrics = calculate_metrics(
        y_test,
        final_probability,
        final_threshold,
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
    print("FINAL HYBRID RESULT")
    print("=" * 100)

    print(
        f"LR component: {lr_name}"
    )

    print(
        f"GB component: {gb_name}"
    )

    print(
        f"LR weight: {blend_best['LRWeight']:.2f}"
    )

    print(
        f"GB weight: {blend_best['GBWeight']:.2f}"
    )

    print(
        f"Threshold: {final_threshold:.2f}"
    )

    for metric in REFERENCE:

        actual = final_metrics[metric]
        canonical = REFERENCE[metric]

        print(
            f"{metric:<12}: "
            f"{actual:.4f} "
            f"(canonical max: "
            f"{canonical:.4f}) "
            +
            (
                "PASS"
                if actual > canonical
                else "MISS"
            )
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
        f"HYBRID vs V9 dominance: "
        f"{v9_dominance}/5"
    )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    comparison = pd.DataFrame(
        [
            {
                "Model": "SH-DAS Hybrid",
                "Type": "Final targeted candidate",
                **final_metrics,
                "Threshold": final_threshold,
                "LRWeight": blend_best["LRWeight"],
                "GBWeight": blend_best["GBWeight"],
                "CanonicalDominance": dominance,
            },
            {
                "Model": "SH-DAS V9",
                "Type": "Current proposed benchmark",
                **V9,
                "Threshold": 0.48,
                "LRWeight": np.nan,
                "GBWeight": np.nan,
                "CanonicalDominance": 2,
            },
            {
                "Model": "Canonical Maximum",
                "Type": "Frozen five-model benchmark",
                **REFERENCE,
                "Threshold": np.nan,
                "LRWeight": np.nan,
                "GBWeight": np.nan,
                "CanonicalDominance": np.nan,
            },
        ]
    )

    comparison_path = (
        REPORT_DIR
        / "hybrid_final_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    model_search_path = (
        REPORT_DIR
        / "hybrid_oof_model_search.csv"
    )

    model_df.to_csv(
        model_search_path,
        index=False,
    )

    summary = {
        "selected_lr": lr_name,
        "selected_gb": gb_name,
        "lr_weight": float(
            blend_best["LRWeight"]
        ),
        "gb_weight": float(
            blend_best["GBWeight"]
        ),
        "threshold": final_threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": final_metrics,
        "canonical_reference": REFERENCE,
        "v9_reference": V9,
        "dominance_count": int(
            dominance
        ),
        "beats_all_five": bool(
            dominance == 5
        ),
        "hybrid_vs_v9_dominance": int(
            v9_dominance
        ),
        "runtime_seconds": float(
            time.time() - start
        ),
    }

    summary_path = (
        REPORT_DIR
        / "hybrid_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "SH-DAS HYBRID FINAL CANDIDATE",
        "=" * 80,
        "",
        f"Logistic Regression component: {lr_name}",
        f"Gradient Boosting component: {gb_name}",
        f"LR weight: {blend_best['LRWeight']:.4f}",
        f"GB weight: {blend_best['GBWeight']:.4f}",
        f"Threshold: {final_threshold:.4f}",
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
            f"Hybrid vs V9 dominance: "
            f"{v9_dominance}/5",
            "",
            "DECISION",
            "-" * 80,
        ]
    )

    if dominance == 5:
        report_lines.append(
            "HYBRID CANDIDATE QUALIFIES "
            "FOR ALL-FIVE METRIC DOMINANCE."
        )
    else:
        report_lines.append(
            "HYBRID CANDIDATE DOES NOT ACHIEVE "
            "ALL-FIVE METRIC DOMINANCE."
        )

    report_path = (
        REPORT_DIR
        / "hybrid_report.txt"
    )

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    bundle_path = (
        REPORT_DIR
        / "sh_das_hybrid_model_bundle.joblib"
    )

    joblib.dump(
        {
            "lr_model": final_lr,
            "gb_model": final_gb,
            "lr_weight": float(
                blend_best["LRWeight"]
            ),
            "gb_weight": float(
                blend_best["GBWeight"]
            ),
            "threshold": final_threshold,
            "seed": SEED,
        },
        bundle_path,
    )

    print("\nSAVED")
    print("-" * 100)
    print(comparison_path)
    print(model_search_path)
    print(summary_path)
    print(report_path)
    print(bundle_path)

    print(
        f"\nRuntime: "
        f"{time.time() - start:.2f} seconds"
    )


if __name__ == "__main__":
    main()
