
# ============================================
# SELF-HEALING ML
# Stage 1: Baseline Customer Churn Classifier
# ============================================

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------
# 1. Define file locations
# --------------------------------------------

DATA_PATH = Path("data/raw/Telco-Customer-Churn.csv")
MODEL_PATH = Path("models/baseline_churn_pipeline.joblib")
REPORT_PATH = Path("reports/baseline_metrics.txt")
MATRIX_PATH = Path("reports/confusion_matrix.png")


# Create output folders if they do not exist
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------
# 2. Load the dataset
# --------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Original shape:", df.shape)


# --------------------------------------------
# 3. Basic cleaning
# --------------------------------------------

# Convert TotalCharges to numeric.
# Invalid or blank values become NaN and are handled
# by the imputer in the preprocessing pipeline.
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Remove customerID because it is an identifier,
# not a meaningful customer-behaviour feature.
df = df.drop(columns=["customerID"])

# Convert target labels:
# Yes -> 1 (churned)
# No  -> 0 (did not churn)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Remove rows where the target is missing, if any.
df = df.dropna(subset=["Churn"])


# --------------------------------------------
# 4. Separate features and target
# --------------------------------------------

X = df.drop(columns=["Churn"])
y = df["Churn"].astype(int)

print("Number of features:", X.shape[1])
print("Target distribution:")
print(y.value_counts())


# --------------------------------------------
# 5. Identify numeric and categorical columns
# --------------------------------------------

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)


# --------------------------------------------
# 6. Create preprocessing pipelines
# --------------------------------------------

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


# --------------------------------------------
# 7. Create the complete ML pipeline
# --------------------------------------------

model = Pipeline(
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


# --------------------------------------------
# 8. Split into training and testing data
# --------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# --------------------------------------------
# 9. Train the model
# --------------------------------------------

print("\nTraining Logistic Regression...")
model.fit(X_train, y_train)

print("Training completed.")


# --------------------------------------------
# 10. Generate predictions
# --------------------------------------------

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]


# --------------------------------------------
# 11. Evaluate the model
# --------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test, y_pred, zero_division=0
)
recall = recall_score(
    y_test, y_pred, zero_division=0
)
f1 = f1_score(
    y_test, y_pred, zero_division=0
)
roc_auc = roc_auc_score(y_test, y_probability)

report = classification_report(
    y_test,
    y_pred,
    target_names=["No Churn", "Churn"],
    zero_division=0,
)

print("\n========== BASELINE RESULTS ==========")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nClassification Report:")
print(report)


# --------------------------------------------
# 12. Save evaluation results
# --------------------------------------------

with open(REPORT_PATH, "w", encoding="utf-8") as file:
    file.write("SELF-HEALING ML - BASELINE RESULTS\n")
    file.write("Model: Logistic Regression\n")
    file.write("Dataset: IBM Telco Customer Churn sample\n")
    file.write("Split: 80% training, 20% testing\n")
    file.write("Random state: 42\n\n")
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall: {recall:.4f}\n")
    file.write(f"F1-score: {f1:.4f}\n")
    file.write(f"ROC-AUC: {roc_auc:.4f}\n\n")
    file.write(report)

print("Metrics saved to:", REPORT_PATH)


# --------------------------------------------
# 13. Create and save confusion matrix
# --------------------------------------------

cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Churn", "Churn"],
)

display.plot()
plt.title("Baseline Logistic Regression - Confusion Matrix")
plt.tight_layout()
plt.savefig(MATRIX_PATH, dpi=200)
plt.close()

print("Confusion matrix saved to:", MATRIX_PATH)


# --------------------------------------------
# 14. Save the complete fitted pipeline
# --------------------------------------------

joblib.dump(model, MODEL_PATH)

print("Trained pipeline saved to:", MODEL_PATH)
print("\nBaseline experiment completed successfully.")