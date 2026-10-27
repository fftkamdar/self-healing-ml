from pathlib import Path
import json
import time
import warnings

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
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

SEED = 42

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "controlled"
REPORT_DIR = (
    ROOT
    / "reports"
    / "controlled_concept_drift"
)

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DATA_PATH = (
    DATA_DIR
    / "concept_drift_telco_like_12000.csv"
)


def sigmoid(x):
    return 1.0 / (
        1.0 + np.exp(-np.clip(x, -30, 30))
    )


def generate_dataset():
    rng = np.random.default_rng(SEED)

    n_train = 8000
    n_adaptation = 2000
    n_test = 2000
    n_total = (
        n_train
        + n_adaptation
        + n_test
    )

    # ---------------------------------------------------------
    # Shared feature distribution
    # ---------------------------------------------------------

    tenure = rng.integers(
        0,
        73,
        n_total,
    )

    monthly_charges = rng.uniform(
        20,
        120,
        n_total,
    )

    support_count = rng.integers(
        0,
        5,
        n_total,
    )

    service_count = rng.integers(
        0,
        10,
        n_total,
    )

    month_to_month = rng.integers(
        0,
        2,
        n_total,
    )

    fiber = rng.integers(
        0,
        2,
        n_total,
    )

    electronic_check = rng.integers(
        0,
        2,
        n_total,
    )

    senior = rng.integers(
        0,
        2,
        n_total,
    )

    partner = rng.integers(
        0,
        2,
        n_total,
    )

    dependents = rng.integers(
        0,
        2,
        n_total,
    )

    gender = rng.integers(
        0,
        2,
        n_total,
    )

    noise = rng.normal(
        0,
        0.25,
        n_total,
    )

    # ---------------------------------------------------------
    # OLD CONCEPT
    #
    # Churn increases with:
    # month-to-month contract
    # high monthly charge
    # low tenure
    # fiber
    # electronic check
    #
    # Churn decreases with support/services.
    # ---------------------------------------------------------

    old_risk = (
        3.0 * month_to_month
        + 0.040
        * (monthly_charges - 70)
        - 0.050
        * (tenure - 36)
        + 1.0 * fiber
        + 0.80 * electronic_check
        + 0.50 * senior
        - 0.70 * support_count
        - 0.25 * partner
        - 0.20 * dependents
        - 0.12 * service_count
        + noise
    )

    # ---------------------------------------------------------
    # NEW CONCEPT
    #
    # Important relationships change:
    #
    # month-to-month effect REVERSES
    # tenure effect REVERSES
    # support effect REVERSES
    #
    # This represents concept drift.
    # ---------------------------------------------------------

    new_risk = (
        -3.0 * month_to_month
        + 0.040
        * (monthly_charges - 70)
        + 0.050
        * (tenure - 36)
        + 1.0 * fiber
        + 0.80 * electronic_check
        + 0.50 * senior
        + 0.70 * support_count
        - 0.25 * partner
        - 0.20 * dependents
        - 0.12 * service_count
        + noise
    )

    phase = np.array(
        [0] * n_train
        + [1] * n_adaptation
        + [2] * n_test
    )

    old_probability = sigmoid(
        old_risk
    )

    new_probability = sigmoid(
        new_risk
    )

    churn = np.zeros(
        n_total,
        dtype=int,
    )

    churn[:n_train] = (
        rng.binomial(
            1,
            old_probability[:n_train],
        )
    )

    churn[n_train:] = (
        rng.binomial(
            1,
            new_probability[n_train:],
        )
    )

    # ---------------------------------------------------------
    # Additional Telco-like feature
    # ---------------------------------------------------------

    total_charges = (
        monthly_charges
        * (tenure + 1)
        + rng.normal(
            0,
            30,
            n_total,
        )
    )

    total_charges = np.maximum(
        total_charges,
        0,
    )

    df = pd.DataFrame(
        {
            "gender": np.where(
                gender == 0,
                "Female",
                "Male",
            ),

            "SeniorCitizen": senior,

            "Partner": np.where(
                partner == 1,
                "Yes",
                "No",
            ),

            "Dependents": np.where(
                dependents == 1,
                "Yes",
                "No",
            ),

            "tenure": tenure,

            "MonthlyCharges": monthly_charges,

            "TotalCharges": total_charges,

            "SupportServiceCount": (
                support_count
            ),

            "ServiceCount": (
                service_count
            ),

            "Contract": np.where(
                month_to_month == 1,
                "Month-to-month",
                "Long-term",
            ),

            "InternetService": np.where(
                fiber == 1,
                "Fiber optic",
                "DSL",
            ),

            "PaymentMethod": np.where(
                electronic_check == 1,
                "Electronic check",
                "Other",
            ),

            "Churn": churn,

            "phase": phase,
        }
    )

    return df


def split_dataset(df):

    train = (
        df[df["phase"] == 0]
        .drop(columns=["phase"])
        .reset_index(drop=True)
    )

    adaptation = (
        df[df["phase"] == 1]
        .drop(columns=["phase"])
        .reset_index(drop=True)
    )

    test = (
        df[df["phase"] == 2]
        .drop(columns=["phase"])
        .reset_index(drop=True)
    )

    return (
        train,
        adaptation,
        test,
    )


def make_preprocessor(X):

    numeric_columns = (
        X.select_dtypes(
            include=[
                "number",
                "bool",
            ]
        )
        .columns
        .tolist()
    )

    categorical_columns = [
        c
        for c in X.columns
        if c not in numeric_columns
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
                StandardScaler()
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
                numeric_columns,
            ),
            (
                "cat",
                categorical_pipe,
                categorical_columns,
            ),
        ]
    )


def calculate_metrics(
    y_true,
    probability,
    threshold=0.50,
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


def train_model(
    model,
    X_train,
    y_train,
):

    preprocessor = make_preprocessor(
        X_train
    )

    pipeline = Pipeline(
        [
            (
                "preprocessor",
                preprocessor,
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


def main():

    start = time.time()

    print("=" * 100)
    print(
        "SH-DAS — CONTROLLED "
        "CONCEPT-DRIFT STRESS TEST"
    )
    print("=" * 100)

    # ---------------------------------------------------------
    # Generate 12,000-row dataset
    # ---------------------------------------------------------

    df = generate_dataset()

    df.to_csv(
        DATA_PATH,
        index=False,
    )

    train, adaptation, test = (
        split_dataset(df)
    )

    X_train = train.drop(
        columns=["Churn"]
    )

    y_train = train[
        "Churn"
    ]

    X_adaptation = adaptation.drop(
        columns=["Churn"]
    )

    y_adaptation = adaptation[
        "Churn"
    ]

    X_test = test.drop(
        columns=["Churn"]
    )

    y_test = test[
        "Churn"
    ]

    print(
        f"\nDataset saved: {DATA_PATH}"
    )

    print(
        f"\nTotal records: {len(df):,}"
    )

    print(
        f"Pre-drift training: {len(train):,}"
    )

    print(
        f"Adaptation window: {len(adaptation):,}"
    )

    print(
        f"Post-drift test: {len(test):,}"
    )

    old_rate = (
        y_train.mean()
    )

    new_rate = (
        y_adaptation.mean()
    )

    print(
        f"\nChurn rate before concept drift: "
        f"{old_rate:.4f}"
    )

    print(
        f"Churn rate after concept drift:  "
        f"{new_rate:.4f}"
    )

    print(
        f"Target-rate change: "
        f"{new_rate - old_rate:+.4f}"
    )

    # ---------------------------------------------------------
    # Frozen conventional models
    # ---------------------------------------------------------

    baseline_models = {
        "Logistic Regression":
            LogisticRegression(
                max_iter=3000,
                random_state=SEED,
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                max_depth=5,
                min_samples_leaf=10,
                random_state=SEED,
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=3,
                random_state=SEED,
                n_jobs=-1,
            ),

        "Gradient Boosting":
            GradientBoostingClassifier(
                n_estimators=180,
                max_depth=2,
                learning_rate=0.05,
                min_samples_leaf=8,
                random_state=SEED,
            ),

        "Support Vector Machine":
            SVC(
                probability=True,
                random_state=SEED,
            ),
    }

    baseline_results = []

    print(
        "\nFROZEN CONVENTIONAL MODELS"
    )

    print(
        "-" * 100
    )

    for name, model in (
        baseline_models.items()
    ):

        print(
            f"Training frozen "
            f"{name} ..."
        )

        pipe = train_model(
            model,
            X_train,
            y_train,
        )

        probability = (
            pipe.predict_proba(
                X_test
            )[:, 1]
        )

        metrics = calculate_metrics(
            y_test,
            probability,
            0.50,
        )

        baseline_results.append(
            {
                "Model": name,
                "Type": "Frozen pre-drift model",
                **metrics,
            }
        )

        print(
            f"{name:<25} "
            f"Accuracy={metrics['Accuracy']:.4f} "
            f"Precision={metrics['Precision']:.4f} "
            f"Recall={metrics['Recall']:.4f} "
            f"F1={metrics['F1-score']:.4f} "
            f"AUC={metrics['ROC-AUC']:.4f}"
        )

    baseline_df = pd.DataFrame(
        baseline_results
    )

    # ---------------------------------------------------------
    # Detect concept drift from adaptation
    # ---------------------------------------------------------

    print(
        "\nSH-DAS DRIFT DETECTION"
    )

    print(
        "-" * 100
    )

    feature_shift = {}

    for col in [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "SupportServiceCount",
        "ServiceCount",
    ]:

        before = (
            X_train[col]
            .astype(float)
        )

        after = (
            X_adaptation[col]
            .astype(float)
        )

        mean_change = (
            after.mean()
            - before.mean()
        )

        feature_shift[col] = (
            float(mean_change)
        )

        print(
            f"{col:<25} "
            f"mean change="
            f"{mean_change:+.4f}"
        )

    # ---------------------------------------------------------
    # Performance check using the frozen LR
    # ---------------------------------------------------------

    monitoring_model = train_model(
        LogisticRegression(
            max_iter=3000,
            random_state=SEED,
        ),
        X_train,
        y_train,
    )

    adaptation_probability = (
        monitoring_model
        .predict_proba(
            X_adaptation
        )[:, 1]
    )

    adaptation_metrics = (
        calculate_metrics(
            y_adaptation,
            adaptation_probability,
            0.50,
        )
    )

    print(
        "\nFrozen-model performance "
        "on adaptation window:"
    )

    for metric, value in (
        adaptation_metrics.items()
    ):
        print(
            f"  {metric:<12}: "
            f"{value:.4f}"
        )

    drift_trigger = (
        (
            abs(
                new_rate
                - old_rate
            )
            >= 0.20
        )
        or
        (
            adaptation_metrics[
                "ROC-AUC"
            ]
            < 0.70
        )
    )

    print(
        f"\nSELF-HEALING TRIGGER: "
        f"{'ACTIVATED' if drift_trigger else 'NOT ACTIVATED'}"
    )

    # ---------------------------------------------------------
    # Adaptive SH-DAS recovery
    # ---------------------------------------------------------

    print(
        "\nSH-DAS ADAPTIVE RECOVERY"
    )

    print(
        "-" * 100
    )

    adaptive_models = {
        "LR": LogisticRegression(
            max_iter=3000,
            random_state=SEED,
        ),

        "GB": GradientBoostingClassifier(
            n_estimators=180,
            max_depth=2,
            learning_rate=0.05,
            min_samples_leaf=8,
            random_state=SEED,
        ),

        "RF": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=3,
            random_state=SEED,
            n_jobs=-1,
        ),
    }

    adaptive_probabilities = []

    adaptive_fitted = {}

    for name, model in (
        adaptive_models.items()
    ):

        print(
            f"Retraining adaptive "
            f"{name} on adaptation data ..."
        )

        pipe = train_model(
            model,
            X_adaptation,
            y_adaptation,
        )

        adaptive_fitted[name] = pipe

        probability = (
            pipe.predict_proba(
                X_test
            )[:, 1]
        )

        adaptive_probabilities.append(
            probability
        )

    adaptive_probability = (
        np.mean(
            np.vstack(
                adaptive_probabilities
            ),
            axis=0,
        )
    )

    # Small operating-point search
    # performed ONLY on the adaptation set.
    adaptation_predictions = []

    for name, pipe in (
        adaptive_fitted.items()
    ):

        p = pipe.predict_proba(
            X_adaptation
        )[:, 1]

        adaptation_predictions.append(
            p
        )

    adaptation_blend = (
        np.mean(
            np.vstack(
                adaptation_predictions
            ),
            axis=0,
        )
    )

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in np.arange(
        0.30,
        0.701,
        0.01,
    ):

        m = calculate_metrics(
            y_adaptation,
            adaptation_blend,
            float(threshold),
        )

        if m["F1-score"] > best_f1:

            best_f1 = m["F1-score"]
            best_threshold = float(
                threshold
            )

    adaptive_metrics = calculate_metrics(
        y_test,
        adaptive_probability,
        best_threshold,
    )

    print(
        f"\nSelected adaptation threshold: "
        f"{best_threshold:.2f}"
    )

    for metric, value in (
        adaptive_metrics.items()
    ):
        print(
            f"{metric:<12}: "
            f"{value:.4f}"
        )

    # ---------------------------------------------------------
    # Compare
    # ---------------------------------------------------------

    reference_max = {
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

    dominance = sum(
        adaptive_metrics[m]
        > reference_max[m]
        for m in reference_max
    )

    comparison = baseline_df.copy()

    comparison = pd.concat(
        [
            comparison,
            pd.DataFrame(
                [
                    {
                        "Model":
                            "SH-DAS Adaptive Recovery",
                        "Type":
                            "Self-healing after concept drift",
                        **adaptive_metrics,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    print(
        "\n" + "=" * 100
    )

    print(
        "FINAL CONTROLLED STRESS-TEST RESULT"
    )

    print(
        "=" * 100
    )

    print(
        "\nFrozen-model maxima:"
    )

    for metric, value in (
        reference_max.items()
    ):
        print(
            f"  {metric:<12}: "
            f"{value:.4f}"
        )

    print(
        "\nSH-DAS Adaptive Recovery:"
    )

    for metric in reference_max:

        value = adaptive_metrics[
            metric
        ]

        reference = reference_max[
            metric
        ]

        print(
            f"  {metric:<12}: "
            f"{value:.4f} "
            f"(frozen max={reference:.4f}) "
            +
            (
                "PASS"
                if value > reference
                else "MISS"
            )
        )

    print(
        f"\nDOMINANCE COUNT: "
        f"{dominance}/5"
    )

    print(
        "SH-DAS BETTER ON ALL FIVE METRICS: "
        + (
            "YES"
            if dominance == 5
            else "NO"
        )
    )

    print(
        f"\nDrift trigger: "
        f"{'ACTIVATED' if drift_trigger else 'NOT ACTIVATED'}"
    )

    # ---------------------------------------------------------
    # Save artifacts
    # ---------------------------------------------------------

    comparison_path = (
        REPORT_DIR
        / "concept_drift_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    drift_path = (
        REPORT_DIR
        / "concept_drift_diagnosis.csv"
    )

    pd.DataFrame(
        [
            {
                "Feature": feature,
                "MeanChange": change,
            }
            for feature, change
            in feature_shift.items()
        ]
    ).to_csv(
        drift_path,
        index=False,
    )

    summary = {
        "dataset": str(
            DATA_PATH
        ),
        "total_records": int(
            len(df)
        ),
        "pre_drift_train": int(
            len(train)
        ),
        "adaptation_window": int(
            len(adaptation)
        ),
        "post_drift_test": int(
            len(test)
        ),
        "old_churn_rate": float(
            old_rate
        ),
        "new_churn_rate": float(
            new_rate
        ),
        "target_rate_change": float(
            new_rate - old_rate
        ),
        "drift_trigger": bool(
            drift_trigger
        ),
        "adaptive_threshold": float(
            best_threshold
        ),
        "frozen_reference_max": reference_max,
        "sh_das_metrics": adaptive_metrics,
        "dominance_count": int(
            dominance
        ),
        "beats_all_five": bool(
            dominance == 5
        ),
        "feature_shift": feature_shift,
        "runtime_seconds": float(
            time.time()
            - start
        ),
    }

    summary_path = (
        REPORT_DIR
        / "concept_drift_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "SH-DAS CONTROLLED CONCEPT-DRIFT STRESS TEST",
        "=" * 80,
        "",
        "IMPORTANT: This is a synthetic controlled benchmark.",
        "It is separate from the real IBM Telco Customer Churn experiment.",
        "",
        f"Total records: {len(df):,}",
        f"Pre-drift training: {len(train):,}",
        f"Adaptation: {len(adaptation):,}",
        f"Post-drift test: {len(test):,}",
        "",
        "CONCEPT DRIFT",
        "-" * 80,
        (
            f"Old churn rate: "
            f"{old_rate:.6f}"
        ),
        (
            f"New churn rate: "
            f"{new_rate:.6f}"
        ),
        (
            f"Target-rate change: "
            f"{new_rate - old_rate:+.6f}"
        ),
        (
            "Self-healing trigger: "
            + (
                "ACTIVATED"
                if drift_trigger
                else "NOT ACTIVATED"
            )
        ),
        "",
        "FINAL RESULTS",
        "-" * 80,
    ]

    for metric in reference_max:

        report_lines.append(
            f"{metric}: "
            f"{adaptive_metrics[metric]:.6f} "
            f"(frozen maximum: "
            f"{reference_max[metric]:.6f})"
        )

    report_lines.extend(
        [
            "",
            f"Dominance count: {dominance}/5",
            (
                "Better on all five: "
                + (
                    "YES"
                    if dominance == 5
                    else "NO"
                )
            ),
            "",
            f"Runtime: "
            f"{time.time() - start:.2f} seconds",
        ]
    )

    report_path = (
        REPORT_DIR
        / "concept_drift_report.txt"
    )

    report_path.write_text(
        "\n".join(
            report_lines
        ),
        encoding="utf-8",
    )

    print(
        "\nSAVED"
    )

    print(
        "-" * 100
    )

    print(DATA_PATH)
    print(comparison_path)
    print(drift_path)
    print(summary_path)
    print(report_path)


if __name__ == "__main__":
    main()
