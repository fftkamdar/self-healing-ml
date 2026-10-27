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

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "proposed_model_v13"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# FROZEN CANONICAL BENCHMARK
REFERENCE = {
    "Accuracy": 0.791341,
    "Precision": 0.639860,
    "Recall": 0.810160,
    "F1-score": 0.624742,
    "ROC-AUC": 0.835577,
}

# Current strongest proposed candidate
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

    raise FileNotFoundError("Telco Customer Churn CSV not found.")


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

    if "tenure" in df.columns and "MonthlyCharges" in df.columns:
        df["TenureChargeInteraction"] = (
            df["tenure"] * df["MonthlyCharges"]
        )

    if "Contract" in df.columns:
        df["IsMonthToMonth"] = (
            df["Contract"].astype(str)
            .eq("Month-to-month")
            .astype(int)
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

    if "Partner" in df.columns and "Dependents" in df.columns:
        df["FamilySupportCount"] = (
            df["Partner"].astype(str).eq("Yes").astype(int)
            + df["Dependents"].astype(str).eq("Yes").astype(int)
        )

    return df


def prepare_data(df):
    y = (
        df["Churn"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
    )

    if y.isna().any():
        raise ValueError("Unexpected Churn values found.")

    X = df.drop(columns=["Churn"])

    return X, y.astype(int)


def make_preprocessor(X):
    numeric = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical = [
        c for c in X.columns
        if c not in numeric
    ]

    numeric_pipe = Pipeline(
        [
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

    categorical_pipe = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
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
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ]
    )


def metrics(probability, y_true, threshold):
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


def objective(m):
    names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
    ]

    ratios = [
        m[name] / REFERENCE[name]
        for name in names
    ]

    margins = [
        m[name] - REFERENCE[name]
        for name in names
    ]

    return (
        int(all(r > 1.0 for r in ratios)),
        float(min(ratios)),
        float(np.mean(ratios)),
        float(np.mean(margins)),
        float(m["F1-score"]),
        float(m["ROC-AUC"]),
    )


def model_configs():
    configs = []

    # Logistic regression family
    configs.append(
        (
            "LR_Balanced",
            "lr",
            {
                "C": 0.5,
                "class_weight": "balanced",
            },
        )
    )

    configs.append(
        (
            "LR_ModerateBalance",
            "lr",
            {
                "C": 0.5,
                "class_weight": {
                    0: 1.0,
                    1: 1.6,
                },
            },
        )
    )

    configs.append(
        (
            "LR_LightBalance",
            "lr",
            {
                "C": 1.0,
                "class_weight": {
                    0: 1.0,
                    1: 1.35,
                },
            },
        )
    )

    # Gradient boosting
    configs.append(
        (
            "GB_Conservative",
            "gb",
            {
                "n_estimators": 220,
                "learning_rate": 0.035,
                "max_depth": 2,
                "min_samples_leaf": 8,
                "subsample": 0.90,
            },
        )
    )

    # XGBoost: deliberately lower positive weighting than V12
    # so that precision/accuracy are not sacrificed for recall.
    configs.append(
        (
            "XGB_LightBalance",
            "xgb",
            {
                "n_estimators": 420,
                "max_depth": 2,
                "learning_rate": 0.035,
                "min_child_weight": 5,
                "subsample": 0.90,
                "colsample_bytree": 0.85,
                "gamma": 0.05,
                "reg_lambda": 6.0,
                "reg_alpha": 0.10,
                "scale_pos_weight": 1.35,
            },
        )
    )

    configs.append(
        (
            "XGB_Balanced",
            "xgb",
            {
                "n_estimators": 380,
                "max_depth": 3,
                "learning_rate": 0.030,
                "min_child_weight": 6,
                "subsample": 0.90,
                "colsample_bytree": 0.85,
                "gamma": 0.05,
                "reg_lambda": 7.0,
                "reg_alpha": 0.10,
                "scale_pos_weight": 1.55,
            },
        )
    )

    return configs


def build_estimator(kind, params):
    if kind == "lr":
        return LogisticRegression(
            max_iter=4000,
            random_state=SEED,
            **params,
        )

    if kind == "gb":
        return GradientBoostingClassifier(
            random_state=SEED,
            **params,
        )

    if kind == "xgb":
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=4,
            random_state=SEED,
            **params,
        )

    raise ValueError(kind)


def main():
    start = time.time()

    print("=" * 100)
    print("SH-DAS V13 — ROBUST BALANCED TABULAR MODEL SEARCH")
    print("=" * 100)

    dataset = find_dataset()

    print(f"\nDataset: {dataset}")

    df = engineer_features(
        pd.read_csv(dataset)
    )

    X, y = prepare_data(df)

    # EXACT SAME SPLIT AS CANONICAL BENCHMARK
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
        [X_train, X_validation],
        axis=0,
    )

    y_development = pd.concat(
        [y_train, y_validation],
        axis=0,
    )

    print(f"\nTrain:       {len(X_train):,}")
    print(f"Validation:  {len(X_validation):,}")
    print(f"Development: {len(X_development):,}")
    print(f"Test:        {len(X_test):,}")

    engineered_features = [
        "AverageMonthlySpend",
        "TenureChargeInteraction",
        "IsMonthToMonth",
        "ServiceCount",
        "FamilySupportCount",
    ]

    print("\nProposed engineered features:")
    for feature in engineered_features:
        print(f"  + {feature}")

    preprocessor = make_preprocessor(
        X_development
    )

    X_dev = preprocessor.fit_transform(
        X_development
    )

    X_test_encoded = preprocessor.transform(
        X_test
    )

    configs = model_configs()

    folds = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    oof_predictions = {}
    config_results = []

    print("\n5-FOLD OOF SEARCH")
    print("-" * 100)

    threshold_grid = np.arange(
        0.20,
        0.801,
        0.01,
    )

    for name, kind, params in configs:

        print(f"\nTraining {name} ...")

        oof = np.zeros(
            len(X_dev),
            dtype=float,
        )

        for fold_no, (
            train_idx,
            valid_idx,
        ) in enumerate(
            folds.split(
                X_dev,
                y_development,
            ),
            start=1,
        ):

            model = build_estimator(
                kind,
                params,
            )

            model.fit(
                X_dev[train_idx],
                y_development.iloc[
                    train_idx
                ],
            )

            oof[valid_idx] = (
                model.predict_proba(
                    X_dev[valid_idx]
                )[:, 1]
            )

            fold_auc = roc_auc_score(
                y_development.iloc[
                    valid_idx
                ],
                oof[valid_idx],
            )

            print(
                f"  Fold {fold_no}: "
                f"AUC={fold_auc:.4f}"
            )

        oof_predictions[name] = oof

        best_record = None

        for threshold in threshold_grid:

            m = metrics(
                oof,
                y_development,
                float(threshold),
            )

            score = objective(m)

            record = {
                "Approach": name,
                "Threshold": float(threshold),
                **m,
                "AllAbove": score[0],
                "MinRatio": score[1],
                "MeanRatio": score[2],
                "MeanMargin": score[3],
            }

            if best_record is None:
                best_record = record
            else:
                old_score = objective(
                    {
                        metric: best_record[metric]
                        for metric in REFERENCE
                    }
                )

                if score > old_score:
                    best_record = record

        config_results.append(
            best_record
        )

        print(
            f"  OOF Accuracy={best_record['Accuracy']:.4f} "
            f"Precision={best_record['Precision']:.4f} "
            f"Recall={best_record['Recall']:.4f} "
            f"F1={best_record['F1-score']:.4f} "
            f"AUC={best_record['ROC-AUC']:.4f}"
        )

    # Identify top 3 base models by OOF objective.
    base_df = pd.DataFrame(
        config_results
    )

    base_df = base_df.sort_values(
        by=[
            "AllAbove",
            "MinRatio",
            "MeanRatio",
            "F1-score",
            "ROC-AUC",
        ],
        ascending=False,
    ).reset_index(drop=True)

    top_models = base_df.head(3)[
        "Approach"
    ].tolist()

    print("\nTOP OOF BASE MODELS")
    print("-" * 100)

    for item in top_models:
        print(f"  + {item}")

    # Deterministic blend search.
    # All weights sum to 1.
    blend_weights = [
        (0.60, 0.25, 0.15),
        (0.50, 0.30, 0.20),
        (0.50, 0.25, 0.25),
        (0.40, 0.40, 0.20),
        (0.40, 0.30, 0.30),
        (0.35, 0.35, 0.30),
        (0.30, 0.40, 0.30),
    ]

    blend_records = []

    if len(top_models) == 3:
        a, b, c = top_models

        print(
            "\nDETERMINISTIC TOP-3 BLEND SEARCH"
        )
        print("-" * 100)

        for weights in blend_weights:

            wa, wb, wc = weights

            blended_oof = (
                wa * oof_predictions[a]
                + wb * oof_predictions[b]
                + wc * oof_predictions[c]
            )

            for threshold in threshold_grid:

                m = metrics(
                    blended_oof,
                    y_development,
                    float(threshold),
                )

                score = objective(m)

                blend_records.append(
                    {
                        "Approach": (
                            f"Blend({a},{b},{c})"
                        ),
                        "ModelA": a,
                        "ModelB": b,
                        "ModelC": c,
                        "WeightA": wa,
                        "WeightB": wb,
                        "WeightC": wc,
                        "Threshold": float(
                            threshold
                        ),
                        **m,
                        "AllAbove": score[0],
                        "MinRatio": score[1],
                        "MeanRatio": score[2],
                        "MeanMargin": score[3],
                    }
                )

    all_search = pd.concat(
        [
            base_df,
            pd.DataFrame(blend_records),
        ],
        ignore_index=True,
    )

    all_search = all_search.sort_values(
        by=[
            "AllAbove",
            "MinRatio",
            "MeanRatio",
            "F1-score",
            "ROC-AUC",
        ],
        ascending=False,
    ).reset_index(drop=True)

    best = all_search.iloc[0]

    print("\n" + "=" * 100)
    print("OOF SELECTED APPROACH")
    print("=" * 100)

    print(
        f"Approach:  {best['Approach']}"
    )
    print(
        f"Threshold: {best['Threshold']:.2f}"
    )

    for metric in REFERENCE:
        print(
            f"{metric:<12}: "
            f"{best[metric]:.4f}"
        )

    selected_name = best["Approach"]

    # Refit final selected base models on full development.
    final_models = {}

    for name, kind, params in configs:
        if name in top_models:
            final_models[name] = (
                build_estimator(
                    kind,
                    params,
                )
            )

            final_models[name].fit(
                X_dev,
                y_development,
            )

    # Determine final probability exactly matching selected approach.
    if selected_name in final_models:
        final_probability = (
            final_models[selected_name]
            .predict_proba(
                X_test_encoded
            )[:, 1]
        )

    elif selected_name.startswith("Blend("):

        a = best["ModelA"]
        b = best["ModelB"]
        c = best["ModelC"]

        final_probability = (
            float(best["WeightA"])
            * final_models[a].predict_proba(
                X_test_encoded
            )[:, 1]
            +
            float(best["WeightB"])
            * final_models[b].predict_proba(
                X_test_encoded
            )[:, 1]
            +
            float(best["WeightC"])
            * final_models[c].predict_proba(
                X_test_encoded
            )[:, 1]
        )

    else:
        raise RuntimeError(
            "Could not reconstruct selected approach."
        )

    selected_threshold = float(
        best["Threshold"]
    )

    final_metrics = metrics(
        final_probability,
        y_test,
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
    print("FINAL V13 RESULT")
    print("=" * 100)

    print(
        f"Selected approach: {selected_name}"
    )
    print(
        f"Threshold: {selected_threshold:.2f}"
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
        f"V13 vs V9 dominance: "
        f"{v9_dominance}/5"
    )

    comparison = pd.DataFrame(
        [
            {
                "Model": "SH-DAS V13",
                "Type": "Proposed",
                **final_metrics,
                "Threshold": selected_threshold,
                "Dominance_vs_canonical": dominance,
                "Dominance_vs_V9": v9_dominance,
            },
            {
                "Model": "SH-DAS V9",
                "Type": "Previous proposed",
                **V9,
                "Threshold": 0.48,
                "Dominance_vs_canonical": np.nan,
                "Dominance_vs_V9": np.nan,
            },
            {
                "Model": "Canonical benchmark maximum",
                "Type": "Frozen five-model reference",
                **REFERENCE,
                "Threshold": np.nan,
                "Dominance_vs_canonical": np.nan,
                "Dominance_vs_V9": np.nan,
            },
        ]
    )

    comparison_path = (
        REPORT_DIR
        / "v13_final_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    oof_path = (
        REPORT_DIR
        / "v13_oof_search.csv"
    )

    all_search.to_csv(
        oof_path,
        index=False,
    )

    summary = {
        "version": "V13",
        "selected_approach": selected_name,
        "selected_threshold": selected_threshold,
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
        "v13_vs_v9_dominance": int(
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
        / "v13_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "SH-DAS V13 FINAL REPORT",
        "=" * 80,
        "",
        f"Selected approach: {selected_name}",
        f"Threshold: {selected_threshold:.2f}",
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
            f"V13 vs V9 dominance: "
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
            f"Runtime: "
            f"{time.time() - start:.2f} seconds",
        ]
    )

    report_path = (
        REPORT_DIR
        / "v13_report.txt"
    )

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    # Save the final selected model components.
    bundle_path = (
        REPORT_DIR
        / "sh_das_v13_model_bundle.joblib"
    )

    joblib.dump(
        {
            "preprocessor": preprocessor,
            "models": final_models,
            "selected_approach": selected_name,
            "threshold": selected_threshold,
            "engineered_features": engineered_features,
            "seed": SEED,
        },
        bundle_path,
    )

    print("\nSAVED")
    print("-" * 100)
    print(comparison_path)
    print(oof_path)
    print(summary_path)
    print(report_path)
    print(bundle_path)


if __name__ == "__main__":
    main()
