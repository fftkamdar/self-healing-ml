
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
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


# ============================================================
# SELF-HEALING ML
# Drift-Triggered Retraining Experiment
# ============================================================

DATA_PATH = Path("data/raw/Telco-Customer-Churn.csv")
BASELINE_MODEL_PATH = Path("models/baseline_churn_pipeline.joblib")
DRIFT_SCORES_PATH = Path("reports/feature_drift_scores.csv")

RETRAINED_MODEL_PATH = Path("models/retrained_churn_pipeline.joblib")
COMPARISON_PATH = Path("reports/retraining_comparison.csv")
DECISION_PATH = Path("reports/retraining_decision.txt")

DRIFT_THRESHOLD = 0.25

RETRAINED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. CHECK THE DRIFT REPORT
# ============================================================

print("\nLoading feature drift scores...")

drift_scores = pd.read_csv(DRIFT_SCORES_PATH)
valid_scores = drift_scores["psi"].dropna()

if valid_scores.empty:
    raise ValueError(
        "No valid PSI scores found. Run src/drift_monitor.py first."
    )

maximum_psi = float(valid_scores.max())

print(f"Maximum PSI: {maximum_psi:.6f}")
print(f"Retraining threshold: {DRIFT_THRESHOLD:.2f}")

if maximum_psi >= DRIFT_THRESHOLD:
    retraining_triggered = True
    print("Decision: HIGH DRIFT - retraining experiment triggered.")
else:
    retraining_triggered = False
    print("Decision: Drift below threshold - retraining not triggered.")


# ============================================================
# 2. LOAD DATA AND RECREATE THE BASELINE SPLIT
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df = df.dropna(subset=["Churn"])

X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

X_train = X_train.copy()
X_test = X_test.copy()

print("Original training rows:", len(X_train))
print("Original test rows:", len(X_test))


# ============================================================
# 3. LOAD THE BASELINE MODEL
# ============================================================

baseline_model = joblib.load(BASELINE_MODEL_PATH)

print("Baseline model loaded successfully.")


# ============================================================
# 4. APPLY THE SAME CONTROLLED DRIFT
# ============================================================

X_shifted = X_test.copy()

X_shifted["tenure"] = (
    X_shifted["tenure"] * 1.5
).clip(upper=72)

X_shifted["MonthlyCharges"] = (
    X_shifted["MonthlyCharges"] * 1.2
).clip(upper=120)

print("Controlled drift applied to tenure and MonthlyCharges.")


# ============================================================
# 5. SPLIT SHIFTED DATA INTO RETRAINING AND EVALUATION SETS
# ============================================================

X_drift_train, X_drift_eval, y_drift_train, y_drift_eval = (
    train_test_split(
        X_shifted,
        y_test,
        test_size=0.50,
        random_state=21,
        stratify=y_test,
    )
)

print("Shifted retraining rows:", len(X_drift_train))
print("Held-out shifted evaluation rows:", len(X_drift_eval))


# ============================================================
# 6. EVALUATE THE BASELINE ON HELD-OUT SHIFTED DATA
# ============================================================

baseline_predictions = baseline_model.predict(X_drift_eval)
baseline_probabilities = baseline_model.predict_proba(
    X_drift_eval
)[:, 1]


def calculate_metrics(y_true, predictions, probabilities):
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_true, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_true, probabilities),
    }


baseline_metrics = calculate_metrics(
    y_drift_eval,
    baseline_predictions,
    baseline_probabilities,
)

print("\nBaseline evaluation on shifted held-out data completed.")


# ============================================================
# 7. RETRAIN ONLY IF THE DRIFT RULE IS TRIGGERED
# ============================================================

if retraining_triggered:

    print("\nPreparing retraining dataset...")

    # Combine original training data with the shifted retraining
    # subset. The held-out evaluation subset is kept separate.
    X_retrain = pd.concat(
        [X_train, X_drift_train],
        ignore_index=True,
    )

    y_retrain = pd.concat(
        [y_train, y_drift_train],
        ignore_index=True,
    )

    numeric_features = X_retrain.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X_retrain.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    retrained_model = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    print("Training retrained model...")

    retrained_model.fit(X_retrain, y_retrain)

    retrained_predictions = retrained_model.predict(
        X_drift_eval
    )

    retrained_probabilities = retrained_model.predict_proba(
        X_drift_eval
    )[:, 1]

    retrained_metrics = calculate_metrics(
        y_drift_eval,
        retrained_predictions,
        retrained_probabilities,
    )

    joblib.dump(
        retrained_model,
        RETRAINED_MODEL_PATH,
    )

    print("Retrained model saved:", RETRAINED_MODEL_PATH)

else:

    retrained_metrics = {
        metric: np.nan for metric in baseline_metrics
    }

    print(
        "\nRetraining skipped because the drift threshold "
        "was not reached."
    )


# ============================================================
# 8. COMPARE RESULTS
# ============================================================

comparison_rows = []

for metric in baseline_metrics:

    comparison_rows.append(
        {
            "metric": metric,
            "baseline_on_shifted_eval": baseline_metrics[metric],
            "retrained_on_shifted_eval": retrained_metrics[metric],
        }
    )

comparison_df = pd.DataFrame(comparison_rows)

comparison_df.to_csv(
    COMPARISON_PATH,
    index=False,
)

print("\n========== MODEL COMPARISON ==========")
print(comparison_df.to_string(index=False))

print("\nComparison saved:", COMPARISON_PATH)


# ============================================================
# 9. SAVE THE RETRAINING DECISION
# ============================================================

with open(DECISION_PATH, "w", encoding="utf-8") as file:

    file.write("SELF-HEALING ML - RETRAINING DECISION\n")
    file.write("=" * 45 + "\n\n")

    file.write(f"Maximum PSI: {maximum_psi:.6f}\n")
    file.write(f"PSI threshold: {DRIFT_THRESHOLD:.2f}\n")

    if retraining_triggered:
        file.write("Decision: Retraining experiment triggered\n")
        file.write(
            "Reason: Maximum feature PSI reached or exceeded "
            "the configured threshold.\n"
        )
        file.write(
            "Training data: Original training set plus the "
            "shifted retraining subset.\n"
        )
        file.write(
            "Evaluation data: Held-out shifted subset not used "
            "during retraining.\n"
        )
    else:
        file.write("Decision: Retraining skipped\n")
        file.write(
            "Reason: Maximum feature PSI was below the threshold.\n"
        )

    file.write(
        "\nThis is a controlled simulation. PSI indicates "
        "distribution change and does not by itself establish "
        "that model performance has degraded. Results should "
        "be validated using representative production data "
        "and confirmed labels before operational deployment.\n"
    )

print("Decision report saved:", DECISION_PATH)
print("\nRetraining experiment completed.")