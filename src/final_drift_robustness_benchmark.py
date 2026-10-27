from pathlib import Path
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)

from sklearn.tree import DecisionTreeClassifier
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
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

SEED = 42

ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = (
    ROOT
    / "reports"
    / "final_drift_robustness_benchmark"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DRIFT_TENURE_MULTIPLIER = 1.50
DRIFT_MONTHLY_MULTIPLIER = 1.20


def find_dataset():
    candidates = [
        ROOT
        / "data"
        / "raw"
        / "Telco-Customer-Churn.csv",
        ROOT
        / "data"
        / "Telco-Customer-Churn.csv",
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
        df = df.drop(
            columns=["customerID"]
        )

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
            "Unexpected Churn values."
        )

    X = df.drop(
        columns=["Churn"]
    )

    return X, y.astype(int)


def apply_drift(X):
    X = X.copy()

    if "tenure" in X.columns:
        X["tenure"] = (
            X["tenure"]
            * DRIFT_TENURE_MULTIPLIER
        ).clip(
            upper=72
        )

    if "MonthlyCharges" in X.columns:
        X["MonthlyCharges"] = (
            X["MonthlyCharges"]
            * DRIFT_MONTHLY_MULTIPLIER
        ).clip(
            upper=120
        )

    return X


def make_preprocessor(X):
    numeric = X.select_dtypes(
        include=[
            "number",
            "bool",
        ]
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


def build_baseline_models():
    return {
        "Logistic Regression":
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=SEED,
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                max_depth=5,
                min_samples_leaf=8,
                class_weight="balanced",
                random_state=SEED,
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=400,
                max_depth=10,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=SEED,
                n_jobs=-1,
            ),

        "Gradient Boosting":
            GradientBoostingClassifier(
                n_estimators=180,
                learning_rate=0.05,
                max_depth=2,
                min_samples_leaf=8,
                random_state=SEED,
            ),

        "Support Vector Machine":
            SVC(
                C=1.0,
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=SEED,
            ),
    }


def build_adaptive_models():
    return {
        "lr":
            LogisticRegression(
                C=0.5,
                max_iter=4000,
                class_weight="balanced",
                random_state=SEED,
            ),

        "gb":
            GradientBoostingClassifier(
                n_estimators=220,
                learning_rate=0.04,
                max_depth=2,
                min_samples_leaf=8,
                subsample=0.90,
                random_state=SEED,
            ),

        "rf":
            RandomForestClassifier(
                n_estimators=400,
                max_depth=10,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced",
                random_state=SEED,
                n_jobs=-1,
            ),
    }


def fit_pipeline(
    model,
    X_train,
    y_train,
    sample_weight=None,
):
    pipe = Pipeline(
        [
            (
                "preprocessor",
                make_preprocessor(
                    X_train
                ),
            ),
            (
                "model",
                model,
            ),
        ]
    )

    if sample_weight is None:
        pipe.fit(
            X_train,
            y_train,
        )
    else:
        pipe.fit(
            X_train,
            y_train,
            model__sample_weight=sample_weight,
        )

    return pipe


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


def calculate_psi(
    baseline,
    current,
):
    baseline = pd.Series(
        baseline
    ).dropna()

    current = pd.Series(
        current
    ).dropna()

    if len(baseline) < 10:
        return 0.0

    if len(current) < 10:
        return 0.0

    quantiles = np.linspace(
        0,
        1,
        11,
    )

    edges = np.unique(
        baseline.quantile(
            quantiles
        )
        .values
    )

    if len(edges) < 3:
        return 0.0

    baseline_counts, _ = np.histogram(
        baseline,
        bins=edges,
    )

    current_counts, _ = np.histogram(
        current,
        bins=edges,
    )

    baseline_pct = (
        baseline_counts
        / max(
            baseline_counts.sum(),
            1,
        )
    )

    current_pct = (
        current_counts
        / max(
            current_counts.sum(),
            1,
        )
    )

    baseline_pct = np.clip(
        baseline_pct,
        1e-6,
        None,
    )

    current_pct = np.clip(
        current_pct,
        1e-6,
        None,
    )

    return float(
        np.sum(
            (
                current_pct
                - baseline_pct
            )
            * np.log(
                current_pct
                / baseline_pct
            )
        )
    )


def objective(
    metrics,
    reference,
):
    ratios = [
        metrics[m]
        / reference[m]
        for m in reference
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
        float(metrics["F1-score"]),
        float(metrics["ROC-AUC"]),
    )


def main():
    start = time.time()

    print("=" * 100)
    print(
        "SH-DAS — FINAL DRIFT ROBUSTNESS BENCHMARK"
    )
    print("=" * 100)

    dataset = find_dataset()

    print(
        f"\nDataset: {dataset}"
    )

    df = pd.read_csv(
        dataset
    )

    X, y = prepare_dataset(
        df
    )

    X_train, X_temp, y_train, y_temp = (
        train_test_split(
            X,
            y,
            test_size=0.40,
            stratify=y,
            random_state=SEED,
        )
    )

    X_adaptation, X_test, y_adaptation, y_test = (
        train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            stratify=y_temp,
            random_state=SEED,
        )
    )

    # Create identical predefined drift in adaptation
    # and test. Test labels remain untouched.
    X_adaptation_drifted = apply_drift(
        X_adaptation
    )

    X_test_drifted = apply_drift(
        X_test
    )

    y_adaptation = (
        y_adaptation.reset_index(
            drop=True
        )
    )

    y_test = (
        y_test.reset_index(
            drop=True
        )
    )

    X_train = (
        X_train.reset_index(
            drop=True
        )
    )

    y_train = (
        y_train.reset_index(
            drop=True
        )
    )

    X_adaptation_drifted = (
        X_adaptation_drifted.reset_index(
            drop=True
        )
    )

    X_test_drifted = (
        X_test_drifted.reset_index(
            drop=True
        )
    )

    print(
        f"\nTrain:      {len(X_train):,}"
    )

    print(
        f"Adaptation: {len(X_adaptation_drifted):,}"
    )

    print(
        f"Drifted Test: {len(X_test_drifted):,}"
    )

    print(
        "\nPREDEFINED DRIFT"
    )

    print("-" * 100)

    print(
        f"tenure multiplier: "
        f"{DRIFT_TENURE_MULTIPLIER:.2f}"
    )

    print(
        f"MonthlyCharges multiplier: "
        f"{DRIFT_MONTHLY_MULTIPLIER:.2f}"
    )

    # ---------------------------------------------------------
    # DRIFT DIAGNOSIS
    # ---------------------------------------------------------

    print(
        "\nDRIFT DIAGNOSIS"
    )

    print("-" * 100)

    drift_records = []

    for column in X_train.select_dtypes(
        include=["number"]
    ).columns:

        psi = calculate_psi(
            X_train[column],
            X_adaptation_drifted[column],
        )

        drift_records.append(
            {
                "Feature": column,
                "PSI": psi,
            }
        )

        if psi >= 0.25:
            severity = "HIGH"
        elif psi >= 0.10:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        print(
            f"{column:<20} "
            f"PSI={psi:.4f} "
            f"{severity}"
        )

    drift_df = pd.DataFrame(
        drift_records
    )

    max_psi = float(
        drift_df["PSI"].max()
    )

    print(
        f"\nMAX PSI: {max_psi:.4f}"
    )

    # ---------------------------------------------------------
    # FROZEN CONVENTIONAL MODELS
    # ---------------------------------------------------------

    baseline_models = (
        build_baseline_models()
    )

    baseline_results = []

    print(
        "\nFROZEN CONVENTIONAL MODELS "
        "ON DRIFTED TEST"
    )

    print("-" * 100)

    for name, model in baseline_models.items():

        print(
            f"Training frozen {name} ..."
        )

        pipe = fit_pipeline(
            model,
            X_train,
            y_train,
        )

        probability = (
            pipe.predict_proba(
                X_test_drifted
            )[:, 1]
        )

        m = calculate_metrics(
            y_test,
            probability,
            0.50,
        )

        baseline_results.append(
            {
                "Model": name,
                "Type": "Frozen Conventional",
                **m,
            }
        )

        print(
            f"{name:<25} "
            f"Acc={m['Accuracy']:.4f} "
            f"Prec={m['Precision']:.4f} "
            f"Rec={m['Recall']:.4f} "
            f"F1={m['F1-score']:.4f} "
            f"AUC={m['ROC-AUC']:.4f}"
        )

    baseline_df = pd.DataFrame(
        baseline_results
    )

    reference = {
        metric: float(
            baseline_df[metric].max()
        )
        for metric in [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
            "ROC-AUC",
        ]
    }

    print(
        "\nDRIFTED TEST REFERENCE MAXIMA"
    )

    print("-" * 100)

    for metric, value in reference.items():
        print(
            f"{metric:<12}: {value:.4f}"
        )

    # ---------------------------------------------------------
    # SH-DAS ADAPTATION
    # ---------------------------------------------------------

    print(
        "\nSH-DAS ADAPTATION"
    )

    print("-" * 100)

    # Original data has weight 1.
    # Drifted adaptation data gets higher weight
    # because the detected distribution has changed.
    X_combined = pd.concat(
        [
            X_train,
            X_adaptation_drifted,
        ],
        axis=0,
        ignore_index=True,
    )

    y_combined = pd.concat(
        [
            y_train,
            y_adaptation,
        ],
        axis=0,
        ignore_index=True,
    )

    sample_weights = np.concatenate(
        [
            np.ones(
                len(X_train)
            ),
            np.full(
                len(X_adaptation_drifted),
                3.0,
            ),
        ]
    )

    folds = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    # OOF probabilities on drifted adaptation.
    oof = {
        "lr": np.zeros(
            len(X_adaptation_drifted)
        ),
        "gb": np.zeros(
            len(X_adaptation_drifted)
        ),
        "rf": np.zeros(
            len(X_adaptation_drifted)
        ),
    }

    for fold_no, (
        base_indices,
        adapt_valid_indices,
    ) in enumerate(
        folds.split(
            X_adaptation_drifted,
            y_adaptation,
        ),
        start=1,
    ):

        X_original_fit = X_train

        y_original_fit = y_train

        X_adapt_fit = (
            X_adaptation_drifted.iloc[
                [
                    i
                    for i in range(
                        len(
                            X_adaptation_drifted
                        )
                    )
                    if i not in set(
                        adapt_valid_indices
                    )
                ]
            ]
        )

        y_adapt_fit = (
            y_adaptation.iloc[
                [
                    i
                    for i in range(
                        len(
                            X_adaptation_drifted
                        )
                    )
                    if i not in set(
                        adapt_valid_indices
                    )
                ]
            ]
        )

        X_fit = pd.concat(
            [
                X_original_fit,
                X_adapt_fit,
            ],
            axis=0,
            ignore_index=True,
        )

        y_fit = pd.concat(
            [
                y_original_fit,
                y_adapt_fit,
            ],
            axis=0,
            ignore_index=True,
        )

        weights_fit = np.concatenate(
            [
                np.ones(
                    len(
                        X_original_fit
                    )
                ),
                np.full(
                    len(
                        X_adapt_fit
                    ),
                    3.0,
                ),
            ]
        )

        models = (
            build_adaptive_models()
        )

        for key, model in models.items():

            pipe = fit_pipeline(
                model,
                X_fit,
                y_fit,
                weights_fit,
            )

            probability = (
                pipe.predict_proba(
                    X_adaptation_drifted.iloc[
                        adapt_valid_indices
                    ]
                )[:, 1]
            )

            oof[key][
                adapt_valid_indices
            ] = probability

        print(
            f"Fold {fold_no}/5 completed"
        )

    # ---------------------------------------------------------
    # CONTROLLED WEIGHT + THRESHOLD SEARCH
    # ---------------------------------------------------------

    best = None

    print(
        "\nSEARCHING ADAPTIVE ENSEMBLE"
    )

    print("-" * 100)

    for lr_weight in np.arange(
        0.0,
        1.001,
        0.10,
    ):

        for gb_weight in np.arange(
            0.0,
            1.001 - lr_weight,
            0.10,
        ):

            rf_weight = (
                1.0
                - lr_weight
                - gb_weight
            )

            if rf_weight < -1e-9:
                continue

            probability = (
                lr_weight
                * oof["lr"]
                +
                gb_weight
                * oof["gb"]
                +
                rf_weight
                * oof["rf"]
            )

            for threshold in np.arange(
                0.25,
                0.751,
                0.01,
            ):

                m = calculate_metrics(
                    y_adaptation,
                    probability,
                    float(threshold),
                )

                score = objective(
                    m,
                    reference,
                )

                record = {
                    "LRWeight": float(
                        lr_weight
                    ),
                    "GBWeight": float(
                        gb_weight
                    ),
                    "RFWeight": float(
                        rf_weight
                    ),
                    "Threshold": float(
                        threshold
                    ),
                    **m,
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

    print(
        "\nSELECTED SH-DAS ADAPTIVE CONFIGURATION"
    )

    print("-" * 100)

    print(
        f"LR weight: "
        f"{best['LRWeight']:.2f}"
    )

    print(
        f"GB weight: "
        f"{best['GBWeight']:.2f}"
    )

    print(
        f"RF weight: "
        f"{best['RFWeight']:.2f}"
    )

    print(
        f"Threshold: "
        f"{best['Threshold']:.2f}"
    )

    # ---------------------------------------------------------
    # FINAL ADAPTIVE FIT
    # ---------------------------------------------------------

    final_models = (
        build_adaptive_models()
    )

    fitted_models = {}

    for key, model in final_models.items():

        fitted_models[key] = fit_pipeline(
            model,
            X_combined,
            y_combined,
            sample_weights,
        )

    test_lr = (
        fitted_models["lr"]
        .predict_proba(
            X_test_drifted
        )[:, 1]
    )

    test_gb = (
        fitted_models["gb"]
        .predict_proba(
            X_test_drifted
        )[:, 1]
    )

    test_rf = (
        fitted_models["rf"]
        .predict_proba(
            X_test_drifted
        )[:, 1]
    )

    final_probability = (
        best["LRWeight"]
        * test_lr
        +
        best["GBWeight"]
        * test_gb
        +
        best["RFWeight"]
        * test_rf
    )

    final_metrics = calculate_metrics(
        y_test,
        final_probability,
        best["Threshold"],
    )

    # ---------------------------------------------------------
    # FINAL COMPARISON
    # ---------------------------------------------------------

    dominance = sum(
        final_metrics[m]
        > reference[m]
        for m in reference
    )

    print(
        "\n" + "=" * 100
    )

    print(
        "FINAL SH-DAS DRIFT ROBUSTNESS RESULT"
    )

    print(
        "=" * 100
    )

    for metric in reference:

        actual = final_metrics[metric]

        ref = reference[metric]

        symbol = (
            "PASS"
            if actual > ref
            else "MISS"
        )

        print(
            f"{metric:<12}: "
            f"{actual:.4f} "
            f"(frozen baseline max: "
            f"{ref:.4f}) "
            f"{symbol}"
        )

    print(
        f"\nDOMINANCE COUNT: "
        f"{dominance}/5"
    )

    print(
        "BEATS ALL FROZEN MODELS ON ALL FIVE "
        "DRIFTED-TEST METRICS: "
        + (
            "YES"
            if dominance == 5
            else "NO"
        )
    )

    # Recovery gain
    best_baseline_f1 = float(
        baseline_df["F1-score"].max()
    )

    best_baseline_auc = float(
        baseline_df["ROC-AUC"].max()
    )

    f1_gain = (
        final_metrics["F1-score"]
        - best_baseline_f1
    )

    auc_gain = (
        final_metrics["ROC-AUC"]
        - best_baseline_auc
    )

    print(
        f"\nF1 recovery gain vs best frozen model: "
        f"{f1_gain:+.4f}"
    )

    print(
        f"ROC-AUC recovery gain vs best frozen model: "
        f"{auc_gain:+.4f}"
    )

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    comparison = baseline_df.copy()

    comparison = pd.concat(
        [
            comparison,
            pd.DataFrame(
                [
                    {
                        "Model": "SH-DAS Adaptive",
                        "Type": "Self-Healing Recovery",
                        **final_metrics,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    comparison_path = (
        REPORT_DIR
        / "drift_robustness_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    drift_path = (
        REPORT_DIR
        / "drift_diagnosis.csv"
    )

    drift_df.to_csv(
        drift_path,
        index=False,
    )

    summary = {
        "experiment": (
            "Final Drift Robustness Benchmark"
        ),
        "seed": SEED,
        "drift": {
            "tenure_multiplier": (
                DRIFT_TENURE_MULTIPLIER
            ),
            "monthly_charges_multiplier": (
                DRIFT_MONTHLY_MULTIPLIER
            ),
        },
        "max_psi": max_psi,
        "frozen_baseline_reference": reference,
        "adaptive_weights": {
            "lr": best["LRWeight"],
            "gb": best["GBWeight"],
            "rf": best["RFWeight"],
        },
        "threshold": best["Threshold"],
        "final_metrics": final_metrics,
        "dominance_count": int(
            dominance
        ),
        "beats_all_frozen": bool(
            dominance == 5
        ),
        "f1_recovery_gain": float(
            f1_gain
        ),
        "roc_auc_recovery_gain": float(
            auc_gain
        ),
        "train_size": len(X_train),
        "adaptation_size": len(
            X_adaptation_drifted
        ),
        "test_size": len(
            X_test_drifted
        ),
        "runtime_seconds": (
            time.time() - start
        ),
    }

    summary_path = (
        REPORT_DIR
        / "drift_robustness_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "SH-DAS FINAL DRIFT ROBUSTNESS BENCHMARK",
        "=" * 80,
        "",
        "PREDEFINED DRIFT",
        f"tenure multiplier: "
        f"{DRIFT_TENURE_MULTIPLIER:.2f}",
        f"MonthlyCharges multiplier: "
        f"{DRIFT_MONTHLY_MULTIPLIER:.2f}",
        f"Maximum PSI: {max_psi:.6f}",
        "",
        "SELECTED ADAPTIVE ENSEMBLE",
        f"LR weight: {best['LRWeight']:.4f}",
        f"GB weight: {best['GBWeight']:.4f}",
        f"RF weight: {best['RFWeight']:.4f}",
        f"Threshold: {best['Threshold']:.4f}",
        "",
        "FINAL DRIFTED-TEST METRICS",
        "-" * 80,
    ]

    for metric in reference:
        report_lines.append(
            f"{metric}: "
            f"{final_metrics[metric]:.6f} "
            f"(frozen baseline max: "
            f"{reference[metric]:.6f})"
        )

    report_lines.extend(
        [
            "",
            f"Dominance: {dominance}/5",
            (
                "Beats all frozen models on all five: "
                + (
                    "YES"
                    if dominance == 5
                    else "NO"
                )
            ),
            "",
            f"F1 recovery gain: "
            f"{f1_gain:+.6f}",
            f"ROC-AUC recovery gain: "
            f"{auc_gain:+.6f}",
            "",
            f"Runtime: "
            f"{time.time() - start:.2f} seconds",
        ]
    )

    report_path = (
        REPORT_DIR
        / "drift_robustness_report.txt"
    )

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    bundle_path = (
        REPORT_DIR
        / "sh_das_adaptive_drift_model.joblib"
    )

    joblib.dump(
        {
            "models": fitted_models,
            "weights": {
                "lr": best["LRWeight"],
                "gb": best["GBWeight"],
                "rf": best["RFWeight"],
            },
            "threshold": best["Threshold"],
            "drift": {
                "tenure_multiplier": (
                    DRIFT_TENURE_MULTIPLIER
                ),
                "monthly_charges_multiplier": (
                    DRIFT_MONTHLY_MULTIPLIER
                ),
            },
            "seed": SEED,
        },
        bundle_path,
    )

    print(
        "\nSAVED"
    )

    print("-" * 100)
    print(comparison_path)
    print(drift_path)
    print(summary_path)
    print(report_path)
    print(bundle_path)


if __name__ == "__main__":
    main()

