
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------
# 1. Project paths and reproducibility
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "optuna_tuned_churn_pipeline.joblib"
REPORT_PATH = PROJECT_ROOT / "reports" / "optuna_tuning_report.txt"

RANDOM_STATE = 42
N_TRIALS = 30


# ---------------------------------------------------------
# 2. Load and prepare the dataset
# ---------------------------------------------------------

def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}\n"
            "Check that Telco-Customer-Churn.csv is inside data/raw/."
        )

    df = pd.read_csv(DATA_PATH)

    # Convert TotalCharges to numeric.
    # Blank or invalid entries become NaN and are imputed later.
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

    # Remove the customer identifier because it is not a useful
    # predictive feature for this experiment.
    df = df.drop(columns=["customerID"])

    # Encode the target as 0 = No Churn, 1 = Churn.
    y = df["Churn"].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError("Unexpected values found in the Churn target.")

    X = df.drop(columns=["Churn"])

    return X, y.astype(int)


# ---------------------------------------------------------
# 3. Build the preprocessing and model pipeline
# ---------------------------------------------------------

def build_pipeline(C, class_weight):
    numeric_features = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = [
        column for column in X_train.columns
        if column not in numeric_features
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    model = LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=3000,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )


# ---------------------------------------------------------
# 4. Create training, validation, and test partitions
# ---------------------------------------------------------

X, y = load_dataset()

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_train_full,
)

print("=" * 65)
print("SELF-HEALING ML: OPTUNA HYPERPARAMETER SEARCH")
print("=" * 65)
print(f"Training rows:   {len(X_train)}")
print(f"Validation rows: {len(X_valid)}")
print(f"Test rows:       {len(X_test)}")
print(f"Number of trials: {N_TRIALS}")


# ---------------------------------------------------------
# 5. Define the Optuna objective
# ---------------------------------------------------------

def objective(trial):
    C = trial.suggest_float("C", 0.001, 100.0, log=True)

    class_weight = trial.suggest_categorical(
        "class_weight",
        [None, "balanced"],
    )

    model = build_pipeline(C, class_weight)
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_valid)[:, 1]

    # Optimize ROC-AUC, which measures ranking quality across
    # classification thresholds.
    score = roc_auc_score(y_valid, probabilities)

    return score


# ---------------------------------------------------------
# 6. Run the search
# ---------------------------------------------------------

optuna.logging.set_verbosity(optuna.logging.WARNING)

study = optuna.create_study(
    direction="maximize",
    study_name="telco_churn_logistic_regression",
)

study.optimize(objective, n_trials=N_TRIALS)

best_params = study.best_params
best_validation_auc = study.best_value

print("\nOptuna search completed.")
print("Best validation ROC-AUC:", round(best_validation_auc, 4))
print("Best parameters:", best_params)


# ---------------------------------------------------------
# 7. Train the selected configuration on all training data
# ---------------------------------------------------------

best_model = build_pipeline(
    C=best_params["C"],
    class_weight=best_params["class_weight"],
)

best_model.fit(X_train_full, y_train_full)


# ---------------------------------------------------------
# 8. Evaluate once on the held-out test set
# ---------------------------------------------------------

test_predictions = best_model.predict(X_test)
test_probabilities = best_model.predict_proba(X_test)[:, 1]

metrics = {
    "Accuracy": accuracy_score(y_test, test_predictions),
    "Precision": precision_score(
        y_test, test_predictions, zero_division=0
    ),
    "Recall": recall_score(
        y_test, test_predictions, zero_division=0
    ),
    "F1-score": f1_score(
        y_test, test_predictions, zero_division=0
    ),
    "ROC-AUC": roc_auc_score(y_test, test_probabilities),
}


# ---------------------------------------------------------
# 9. Save model and experiment report
# ---------------------------------------------------------

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(best_model, MODEL_PATH)

report_lines = [
    "SELF-HEALING ML - OPTUNA TUNING REPORT",
    "=" * 50,
    f"Dataset: {DATA_PATH.name}",
    f"Total rows: {len(X)}",
    f"Optuna trials: {N_TRIALS}",
    "Optimization objective: Validation ROC-AUC",
    f"Best validation ROC-AUC: {best_validation_auc:.4f}",
    "",
    "Best hyperparameters:",
]

for name, value in best_params.items():
    report_lines.append(f"- {name}: {value}")

report_lines.extend(
    [
        "",
        "Held-out test metrics:",
    ]
)

for name, value in metrics.items():
    report_lines.append(f"- {name}: {value:.4f}")

report_lines.extend(
    [
        "",
        "Classification report:",
        classification_report(
            y_test,
            test_predictions,
            target_names=["No Churn", "Churn"],
            zero_division=0,
        ),
        "",
        "Methodology notes:",
        "- The test set was held out from hyperparameter selection.",
        "- Preprocessing was fitted within each training pipeline.",
        "- This experiment uses the original dataset, not live data.",
        "- Hyperparameter tuning does not guarantee improved performance.",
    ]
)

REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

print("\nHELD-OUT TEST RESULTS")
print("-" * 40)

for name, value in metrics.items():
    print(f"{name:10s}: {value:.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        test_predictions,
        target_names=["No Churn", "Churn"],
        zero_division=0,
    )
)

print("Tuned model saved to:", MODEL_PATH)
print("Tuning report saved to:", REPORT_PATH)
print("\nOptuna tuning experiment completed successfully.")