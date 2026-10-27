
from pathlib import Path
import time

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


# ---------------------------------------------------------
# 1. PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "benchmark"
REPORT_DIR = PROJECT_ROOT / "reports" / "benchmark"

RESULTS_CSV = REPORT_DIR / "five_model_comparison.csv"
CHART_PATH = REPORT_DIR / "five_model_comparison.png"
CONFUSION_CHART = REPORT_DIR / "five_model_confusion_matrices.png"


# ---------------------------------------------------------
# 2. LOAD AND CLEAN DATA
# ---------------------------------------------------------

def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Check that Telco-Customer-Churn.csv is inside data/raw."
        )

    df = pd.read_csv(DATA_PATH)

    # Remove the customer identifier because it is not a
    # meaningful predictive feature.
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Convert TotalCharges to numeric.
    # Invalid or blank values become NaN and are imputed later.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"], errors="coerce"
        )

    if "Churn" not in df.columns:
        raise ValueError("The dataset must contain a 'Churn' target column.")

    # Keep only rows with a known target.
    df = df.dropna(subset=["Churn"]).copy()

    # Encode the target: Yes = 1, No = 0.
    y = df["Churn"].map({"Yes": 1, "No": 0})

    if y.isna().any():
        raise ValueError(
            "Unexpected values found in Churn. Expected 'Yes' and 'No'."
        )

    X = df.drop(columns=["Churn"])

    return X, y.astype(int)


# ---------------------------------------------------------
# 3. PREPROCESSING PIPELINE
# ---------------------------------------------------------

def create_preprocessor(X):
    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
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
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor


# ---------------------------------------------------------
# 4. DEFINE FIVE CLASSIFICATION MODELS
# ---------------------------------------------------------

def create_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),

        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        ),

        "Support Vector Machine": SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            class_weight="balanced",
            random_state=42,
        ),
    }


# ---------------------------------------------------------
# 5. TRAIN AND EVALUATE THE MODELS
# ---------------------------------------------------------

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset()

    # Use one common stratified split for every model.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = create_models()
    rows = []
    fitted_models = {}
    predictions = {}

    print("=" * 75)
    print("FIVE-MODEL CHURN CLASSIFICATION BENCHMARK")
    print("=" * 75)
    print(f"Dataset rows: {len(X)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Number of models: {len(models)}")

    for model_name, classifier in models.items():
        print(f"\nTraining: {model_name}")

        # Create a separate preprocessing pipeline for each model.
        pipeline = Pipeline(
            steps=[
                ("preprocessor", create_preprocessor(X_train)),
                ("classifier", classifier),
            ]
        )

        start_time = time.perf_counter()
        pipeline.fit(X_train, y_train)
        training_time = time.perf_counter() - start_time

        y_pred = pipeline.predict(X_test)
        y_probability = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(
                y_test, y_pred, zero_division=0
            ),
            "Recall": recall_score(
                y_test, y_pred, zero_division=0
            ),
            "F1-score": f1_score(
                y_test, y_pred, zero_division=0
            ),
            "ROC-AUC": roc_auc_score(y_test, y_probability),
            "Training_Time_Seconds": training_time,
        }

        rows.append(metrics)
        fitted_models[model_name] = pipeline
        predictions[model_name] = y_pred

        # Save each trained model separately.
        safe_name = (
            model_name.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )
        model_path = MODEL_DIR / f"{safe_name}.joblib"
        joblib.dump(pipeline, model_path)

        print(
            f"Accuracy:  {metrics['Accuracy']:.4f}\n"
            f"Precision: {metrics['Precision']:.4f}\n"
            f"Recall:    {metrics['Recall']:.4f}\n"
            f"F1-score:  {metrics['F1-score']:.4f}\n"
            f"ROC-AUC:   {metrics['ROC-AUC']:.4f}\n"
            f"Train time:{training_time:.2f} seconds"
        )

    # -----------------------------------------------------
    # 6. SAVE COMPARISON RESULTS
    # -----------------------------------------------------

    results = pd.DataFrame(rows)
    results = results.sort_values(
        by="ROC-AUC", ascending=False
    ).reset_index(drop=True)

    results.to_csv(RESULTS_CSV, index=False)

    print("\n" + "=" * 75)
    print("FIVE-MODEL COMPARISON RESULTS")
    print("=" * 75)
    print(
        results.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # -----------------------------------------------------
    # 7. CREATE METRIC COMPARISON CHART
    # -----------------------------------------------------

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
    ]

    chart_data = results.set_index("Model")[metric_columns]

    ax = chart_data.plot(
        kind="bar",
        figsize=(13, 7),
        width=0.8,
    )

    ax.set_title("Performance Comparison of Five ML Models")
    ax.set_ylabel("Score")
    ax.set_xlabel("Machine Learning Model")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=300)
    plt.close()

    # -----------------------------------------------------
    # 8. CREATE CONFUSION MATRICES
    # -----------------------------------------------------

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for index, model_name in enumerate(models.keys()):
        ConfusionMatrixDisplay(
            confusion_matrix(
                y_test, predictions[model_name], labels=[0, 1]
            ),
            display_labels=["No Churn", "Churn"],
        ).plot(
            ax=axes[index],
            colorbar=False,
            values_format="d",
        )

        axes[index].set_title(model_name)

    # Hide the unused sixth panel.
    axes[-1].axis("off")

    fig.suptitle("Confusion Matrices — Five-Model Benchmark")
    plt.tight_layout()
    plt.savefig(CONFUSION_CHART, dpi=300)
    plt.close()

    print("\nSaved benchmark artifacts:")
    print(f"Comparison CSV: {RESULTS_CSV}")
    print(f"Metric chart:   {CHART_PATH}")
    print(f"Confusion plot: {CONFUSION_CHART}")
    print(f"Saved models in: {MODEL_DIR}")

    print("\nBenchmark completed.")
    print(
        "Reminder: this benchmark does not automatically replace "
        "the selected model in the self-healing pipeline."
    )


if __name__ == "__main__":
    main()