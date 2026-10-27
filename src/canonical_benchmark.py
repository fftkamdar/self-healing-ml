from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)

from sklearn.tree import DecisionTreeClassifier
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
REPORT_DIR = ROOT / "reports" / "canonical_benchmark"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


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

    raise FileNotFoundError("Telco CSV not found.")


def prepare_dataset(df):
    df = df.copy()

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

    y = (
        df["Churn"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
    )

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
                SimpleImputer(strategy="median")
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
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            ),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ]
    )


def evaluate(y_true, probability):
    prediction = (
        probability >= 0.50
    ).astype(int)

    return {
        "Accuracy": accuracy_score(
            y_true,
            prediction
        ),
        "Precision": precision_score(
            y_true,
            prediction,
            zero_division=0
        ),
        "Recall": recall_score(
            y_true,
            prediction,
            zero_division=0
        ),
        "F1-score": f1_score(
            y_true,
            prediction,
            zero_division=0
        ),
        "ROC-AUC": roc_auc_score(
            y_true,
            probability
        ),
    }


def main():
    start = time.time()

    print("=" * 100)
    print("SH-DAS — CANONICAL FIVE-MODEL BENCHMARK")
    print("=" * 100)

    dataset = find_dataset()

    print(f"\nDataset: {dataset}")

    df = pd.read_csv(dataset)

    X, y = prepare_dataset(df)

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

    print("\nFeatures:")
    for col in X.columns:
        print(f"  + {col}")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=SEED,
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=SEED,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=8,
            random_state=SEED,
        ),

        "Support Vector Machine": SVC(
            C=1.0,
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=SEED,
        ),
    }

    results = []

    print("\nTRAINING")
    print("-" * 100)

    for name, model in models.items():

        print(f"Training {name} ...")

        preprocessor = make_preprocessor(
            X_train
        )

        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(
            X_train,
            y_train
        )

        test_probability = (
            pipeline.predict_proba(X_test)[:, 1]
        )

        metrics = evaluate(
            y_test,
            test_probability
        )

        results.append(
            {
                "Model": name,
                "Accuracy": metrics["Accuracy"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1-score": metrics["F1-score"],
                "ROC-AUC": metrics["ROC-AUC"],
            }
        )

        print(
            f"{name:<25}"
            f" Accuracy={metrics['Accuracy']:.4f}"
            f" Precision={metrics['Precision']:.4f}"
            f" Recall={metrics['Recall']:.4f}"
            f" F1={metrics['F1-score']:.4f}"
            f" AUC={metrics['ROC-AUC']:.4f}"
        )

    result_df = pd.DataFrame(results)

    maxima = {
        metric: float(
            result_df[metric].max()
        )
        for metric in [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
            "ROC-AUC",
        ]
    }

    print("\n" + "=" * 100)
    print("CANONICAL BENCHMARK")
    print("=" * 100)

    print(result_df.to_string(index=False))

    print("\nCANONICAL METRIC MAXIMA")
    print("-" * 100)

    for metric, value in maxima.items():
        model = result_df.loc[
            result_df[metric].idxmax(),
            "Model"
        ]

        print(
            f"{metric:<12}: "
            f"{value:.4f} "
            f"← {model}"
        )

    comparison_path = (
        REPORT_DIR
        / "canonical_five_model_benchmark.csv"
    )

    result_df.to_csv(
        comparison_path,
        index=False
    )

    maxima_path = (
        REPORT_DIR
        / "canonical_metric_maxima.json"
    )

    maxima_path.write_text(
        json.dumps(
            maxima,
            indent=2
        ),
        encoding="utf-8"
    )

    metadata = {
        "seed": SEED,
        "train_size": len(X_train),
        "validation_size": len(X_validation),
        "test_size": len(X_test),
        "feature_count": len(X.columns),
        "features": list(X.columns),
        "dataset": str(dataset),
        "runtime_seconds": time.time() - start,
    }

    metadata_path = (
        REPORT_DIR
        / "canonical_benchmark_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\nSaved:")
    print(comparison_path)
    print(maxima_path)
    print(metadata_path)

    print(
        f"\nRuntime: "
        f"{time.time() - start:.2f} seconds"
    )


if __name__ == "__main__":
    main()


