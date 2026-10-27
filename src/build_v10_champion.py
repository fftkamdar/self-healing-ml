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
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")

SEED = 42

ROOT = Path(__file__).resolve().parents[1]

DATA_CANDIDATES = [
    ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    ROOT / "data" / "Telco-Customer-Churn.csv",
    ROOT / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    ROOT / "Telco-Customer-Churn.csv",
]

REPORT_DIR = ROOT / "reports" / "proposed_model_v10"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def find_dataset():
    for path in DATA_CANDIDATES:
        if path.exists():
            return path

    matches = list(ROOT.rglob("*.csv"))
    for path in matches:
        name = path.name.lower()
        if "telco" in name or "churn" in name:
            return path

    raise FileNotFoundError("Telco Customer Churn CSV could not be found.")


def engineer_features(df):
    df = df.copy()

    # Basic cleaning
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"], errors="coerce"
        )

    # Feature engineering
    if "tenure" in df.columns and "TotalCharges" in df.columns:
        df["AverageMonthlySpend"] = (
            df["TotalCharges"] /
            df["tenure"].replace(0, np.nan)
        )
        df["AverageMonthlySpend"] = df["AverageMonthlySpend"].replace(
            [np.inf, -np.inf], np.nan
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

    existing_services = [c for c in service_columns if c in df.columns]

    if existing_services:
        service_count = 0
        for col in existing_services:
            service_count += (
                ~df[col].astype(str).isin(["No", "No internet service", "nan"])
            ).astype(int)
        df["ServiceCount"] = service_count

    if "Contract" in df.columns:
        df["IsMonthToMonth"] = (
            df["Contract"].astype(str)
            .eq("Month-to-month")
            .astype(int)
        )

    family_columns = [
        "Partner",
        "Dependents",
    ]

    existing_family = [c for c in family_columns if c in df.columns]

    if existing_family:
        df["FamilySupportCount"] = 0
        for col in existing_family:
            df["FamilySupportCount"] += (
                df[col].astype(str).eq("Yes")
            ).astype(int)

    # Explicit interaction-style numerical features
    if "tenure" in df.columns and "MonthlyCharges" in df.columns:
        df["TenureChargeInteraction"] = (
            df["tenure"] * df["MonthlyCharges"]
        )

    return df


def prepare_target(df):
    if "Churn" not in df.columns:
        raise ValueError("Target column 'Churn' was not found.")

    y = (
        df["Churn"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
    )

    if y.isna().any():
        raise ValueError("Unexpected values found in Churn column.")

    X = df.drop(columns=["Churn"])

    return X, y.astype(int)


def make_preprocessor(X):
    numeric_cols = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_cols = [
        c for c in X.columns if c not in numeric_cols
    ]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
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
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def metric_dict(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "Accuracy": float(accuracy_score(y_true, predictions)),
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


def score_against_reference(metrics, reference):
    metric_names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
    ]

    dominance = sum(
        metrics[m] > reference[m]
        for m in metric_names
    )

    margins = [
        metrics[m] - reference[m]
        for m in metric_names
    ]

    mean_margin = float(np.mean(margins))

    balanced_score = (
        0.15 * metrics["Accuracy"]
        + 0.15 * metrics["Precision"]
        + 0.15 * metrics["Recall"]
        + 0.35 * metrics["F1-score"]
        + 0.20 * metrics["ROC-AUC"]
    )

    return dominance, mean_margin, balanced_score


def build_candidates(X_train):
    candidates = []

    candidates.append(
        (
            "LogisticRegression_Engineered",
            LogisticRegression(
                C=1.0,
                max_iter=3000,
                class_weight="balanced",
                random_state=SEED,
            ),
        )
    )

    candidates.append(
        (
            "LogisticRegression_StrongerRegularization",
            LogisticRegression(
                C=0.35,
                max_iter=3000,
                class_weight="balanced",
                random_state=SEED,
            ),
        )
    )

    candidates.append(
        (
            "GradientBoosting_Engineered",
            GradientBoostingClassifier(
                n_estimators=250,
                learning_rate=0.04,
                max_depth=2,
                min_samples_leaf=8,
                subsample=0.90,
                random_state=SEED,
            ),
        )
    )

    candidates.append(
        (
            "GradientBoosting_Wider",
            GradientBoostingClassifier(
                n_estimators=180,
                learning_rate=0.05,
                max_depth=3,
                min_samples_leaf=10,
                subsample=0.90,
                random_state=SEED,
            ),
        )
    )

    candidates.append(
        (
            "RandomForest_Engineered",
            RandomForestClassifier(
                n_estimators=450,
                max_depth=10,
                min_samples_leaf=3,
                max_features="sqrt",
                class_weight="balanced",
                random_state=SEED,
                n_jobs=-1,
            ),
        )
    )

    candidates.append(
        (
            "ExtraTrees_Engineered",
            ExtraTreesClassifier(
                n_estimators=450,
                max_depth=14,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced",
                random_state=SEED,
                n_jobs=-1,
            ),
        )
    )

    calibrated_svm = CalibratedClassifierCV(
        LinearSVC(
            C=0.35,
            class_weight="balanced",
            random_state=SEED,
            max_iter=5000,
        ),
        method="sigmoid",
        cv=3,
    )

    candidates.append(
        (
            "CalibratedLinearSVM",
            calibrated_svm,
        )
    )

    return candidates


def main():
    start = time.time()

    print("=" * 90)
    print("SH-DAS V10 — FEATURE-ENGINEERED PERFORMANCE SEARCH")
    print("=" * 90)

    dataset_path = find_dataset()

    print(f"\nDataset: {dataset_path}")

    df = pd.read_csv(dataset_path)
    df = engineer_features(df)

    X, y = prepare_target(df)

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

    print("\nFEATURES")
    print("-" * 90)
    engineered = [
        c for c in X.columns
        if c in [
            "AverageMonthlySpend",
            "ServiceCount",
            "IsMonthToMonth",
            "FamilySupportCount",
            "TenureChargeInteraction",
        ]
    ]

    print("Engineered features:")
    for col in engineered:
        print(f"  + {col}")

    # Existing benchmark values from the established five-model experiment.
    # These are used ONLY for validation-stage candidate selection.
    reference = {
        "Accuracy": 0.7999,
        "Precision": 0.6608,
        "Recall": 0.7914,
        "F1-score": 0.6143,
        "ROC-AUC": 0.8420,
    }

    candidates = build_candidates(X_train)

    threshold_grid = np.arange(0.20, 0.801, 0.02)

    validation_results = []
    fitted_candidates = {}

    print("\nMODEL EVALUATION")
    print("-" * 90)

    for name, estimator in candidates:
        print(f"Training {name} ...")

        preprocessor = make_preprocessor(X_train)

        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )

        pipe.fit(X_train, y_train)

        validation_prob = pipe.predict_proba(X_validation)[:, 1]

        best_for_candidate = None

        for threshold in threshold_grid:
            metrics = metric_dict(
                y_validation,
                validation_prob,
                float(threshold),
            )

            dominance, mean_margin, balanced_score = (
                score_against_reference(
                    metrics,
                    reference,
                )
            )

            record = {
                "Model": name,
                "Threshold": float(threshold),
                **metrics,
                "Dominance": int(dominance),
                "MeanMargin": float(mean_margin),
                "BalancedScore": float(balanced_score),
            }

            if best_for_candidate is None:
                best_for_candidate = record
            else:
                old = (
                    best_for_candidate["Dominance"],
                    best_for_candidate["MeanMargin"],
                    best_for_candidate["BalancedScore"],
                )
                new = (
                    record["Dominance"],
                    record["MeanMargin"],
                    record["BalancedScore"],
                )

                if new > old:
                    best_for_candidate = record

        fitted_candidates[name] = (
            pipe,
            float(best_for_candidate["Threshold"]),
        )

        validation_results.append(best_for_candidate)

        print(
            f"{name:<40} "
            f"ValF1={best_for_candidate['F1-score']:.4f} "
            f"ValAUC={best_for_candidate['ROC-AUC']:.4f} "
            f"Threshold={best_for_candidate['Threshold']:.2f} "
            f"Dom={best_for_candidate['Dominance']}/5"
        )

    validation_df = pd.DataFrame(validation_results)

    validation_df = validation_df.sort_values(
        by=[
            "Dominance",
            "MeanMargin",
            "BalancedScore",
        ],
        ascending=False,
    ).reset_index(drop=True)

    print("\nVALIDATION LEADER")
    print("-" * 90)
    print(validation_df.iloc[0].to_string())

    selected_name = validation_df.iloc[0]["Model"]
    selected_threshold = float(
        validation_df.iloc[0]["Threshold"]
    )

    print("\nSELECTED CANDIDATE")
    print("-" * 90)
    print(f"Model:     {selected_name}")
    print(f"Threshold: {selected_threshold:.2f}")

    # Refit selected architecture on train + validation
    X_train_full = pd.concat(
        [X_train, X_validation],
        axis=0,
    )
    y_train_full = pd.concat(
        [y_train, y_validation],
        axis=0,
    )

    selected_estimator = None

    for name, estimator in candidates:
        if name == selected_name:
            selected_estimator = estimator
            break

    final_preprocessor = make_preprocessor(X_train_full)

    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", final_preprocessor),
            ("model", selected_estimator),
        ]
    )

    print("\nREFITTING SELECTED MODEL ON TRAIN + VALIDATION ...")
    final_pipeline.fit(X_train_full, y_train_full)

    test_prob = final_pipeline.predict_proba(X_test)[:, 1]

    final_metrics = metric_dict(
        y_test,
        test_prob,
        selected_threshold,
    )

    existing_reference = {
        "Accuracy": 0.8013,
        "Precision": 0.6679,
        "Recall": 0.7968,
        "F1-score": 0.6179,
        "ROC-AUC": 0.8430,
    }

    dominance, mean_margin, balanced_score = (
        score_against_reference(
            final_metrics,
            existing_reference,
        )
    )

    comparison_rows = []

    comparison_rows.append(
        {
            "Model": selected_name,
            "Type": "SH-DAS V10",
            **final_metrics,
            "Threshold": selected_threshold,
        }
    )

    comparison_rows.append(
        {
            "Model": "Existing Best Across Five",
            "Type": "Existing reference",
            **existing_reference,
            "Threshold": np.nan,
        }
    )

    comparison_df = pd.DataFrame(comparison_rows)

    comparison_path = (
        REPORT_DIR / "v10_final_comparison.csv"
    )
    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    validation_path = (
        REPORT_DIR / "v10_validation_results.csv"
    )
    validation_df.to_csv(
        validation_path,
        index=False,
    )

    summary = {
        "version": "V10",
        "selected_model": selected_name,
        "selected_threshold": selected_threshold,
        "engineered_features": engineered,
        "test_metrics": final_metrics,
        "existing_reference": existing_reference,
        "dominance_count": int(dominance),
        "beats_all_five": bool(dominance == 5),
        "mean_margin": float(mean_margin),
        "balanced_score": float(balanced_score),
        "train_size": int(len(X_train)),
        "validation_size": int(len(X_validation)),
        "test_size": int(len(X_test)),
        "runtime_seconds": float(time.time() - start),
    }

    summary_path = (
        REPORT_DIR / "v10_summary.json"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "SH-DAS V10 — FINAL REPORT",
        "=" * 80,
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
            f"{metric}: {final_metrics[metric]:.6f} "
            f"(existing reference: "
            f"{existing_reference[metric]:.6f})"
        )

    report_lines.extend(
        [
            "",
            f"DOMINANCE COUNT: {dominance}/5",
            f"BEATS ALL FIVE ON ALL FIVE METRICS: "
            f"{'YES' if dominance == 5 else 'NO'}",
            "",
            "ENGINEERED FEATURES",
            "-" * 80,
        ]
    )

    report_lines.extend(
        [f"- {x}" for x in engineered]
    )

    report_lines.extend(
        [
            "",
            f"Runtime: {time.time() - start:.2f} seconds",
        ]
    )

    report_path = REPORT_DIR / "v10_report.txt"
    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    bundle_path = (
        REPORT_DIR / "sh_das_v10_model_bundle.joblib"
    )

    joblib.dump(
        {
            "pipeline": final_pipeline,
            "threshold": selected_threshold,
            "selected_model": selected_name,
            "engineered_features": engineered,
            "seed": SEED,
        },
        bundle_path,
    )

    print("\n" + "=" * 90)
    print("FINAL V10 RESULT")
    print("=" * 90)

    print(f"Selected model: {selected_name}")
    print(f"Threshold: {selected_threshold:.2f}")

    for metric in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
    ]:
        actual = final_metrics[metric]
        ref = existing_reference[metric]
        symbol = "✓" if actual > ref else "✗"

        print(
            f"{metric:<12}: "
            f"{actual:.4f} "
            f"(existing max: {ref:.4f}) {symbol}"
        )

    print(f"\nDOMINANCE COUNT: {dominance}/5")
    print(
        "BEATS ALL FIVE ON ALL FIVE METRICS: "
        + ("YES" if dominance == 5 else "NO")
    )

    print("\nSaved:")
    print(comparison_path)
    print(validation_path)
    print(summary_path)
    print(report_path)
    print(bundle_path)


if __name__ == "__main__":
    main()
