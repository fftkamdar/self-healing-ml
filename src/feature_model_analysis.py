"""
Feature-Level Model Analysis
----------------------------

Purpose:
1. Train the same five benchmark models used in the project.
2. Measure permutation importance for every ORIGINAL dataset feature.
3. Identify features with meaningful vs low/unclear contribution.
4. Produce CSV reports and visualizations for dashboard integration.

Important:
"Low/Unclear" does NOT mean that a model mathematically cannot use
the feature. It means that, in this experiment, permuting that feature
did not produce a reliably positive ROC-AUC contribution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"

REPORT_DIR = PROJECT_ROOT / "reports" / "model_feature_analysis"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_REPEATS = 10


# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================

def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load Telco churn data and perform basic project preprocessing."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}\n\n"
            "Check that Telco-Customer-Churn.csv is inside data/raw."
        )

    df = pd.read_csv(DATA_PATH)

    # Convert TotalCharges to numeric.
    # Invalid/blank entries become NaN and are handled by the pipeline.
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce",
    )

    # Remove identifier.
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

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
        raise ValueError(
            "Unexpected values were found in the Churn target column."
        )

    X = df.drop(columns=["Churn"])

    return X, y.astype(int)


# =============================================================================
# PREPROCESSOR
# =============================================================================

def create_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create the same general preprocessing approach as the benchmark."""

    numeric_features = X.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", __import__("sklearn").impute.SimpleImputer(
                strategy="median"
            )),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", __import__("sklearn").impute.SimpleImputer(
                strategy="most_frequent"
            )),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

def build_models() -> Dict[str, object]:
    """Return the five benchmark algorithms."""

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
        "Support Vector Machine": SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def analyze_model(
    model_name: str,
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Train one model and calculate original-feature permutation importance."""

    print(f"\n{'=' * 70}")
    print(f"Analyzing: {model_name}")
    print(f"{'=' * 70}")

    preprocessor = create_preprocessor(X_train)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    print("Training model...")
    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    baseline_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    print(f"Baseline test ROC-AUC: {baseline_auc:.6f}")
    print("Calculating feature-level permutation importance...")

    # IMPORTANT:
    # X_test remains in its ORIGINAL feature space.
    # Permutation importance therefore measures the effect of the
    # original dataset feature rather than each one-hot encoded column.
    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    feature_names = X_test.columns.tolist()

    importance_mean = result.importances_mean
    importance_std = result.importances_std

    output = pd.DataFrame(
        {
            "Model": model_name,
            "Feature": feature_names,
            "Importance_Mean": importance_mean,
            "Importance_Std": importance_std,
        }
    )

    # A conservative interpretation:
    # If mean - std > 0, the importance is reliably positive
    # across the repeated permutations.
    output["Lower_Bound"] = (
        output["Importance_Mean"] -
        output["Importance_Std"]
    )

    output["Feature_Status"] = np.where(
        output["Lower_Bound"] > 0,
        "Meaningful Contribution",
        "Low/Unclear Contribution",
    )

    output["Absolute_Importance"] = output["Importance_Mean"].abs()

    output = output.sort_values(
        by="Absolute_Importance",
        ascending=False,
    )

    print("\nTop features:")
    print(
        output[
            [
                "Feature",
                "Importance_Mean",
                "Importance_Std",
                "Feature_Status",
            ]
        ].head(10).to_string(index=False)
    )

    return output


# =============================================================================
# REPORT GENERATION
# =============================================================================

def create_reports(all_results: pd.DataFrame) -> None:
    """Create all CSV summaries and a JSON report."""

    # -------------------------------------------------------------------------
    # Full long-format results
    # -------------------------------------------------------------------------

    full_path = REPORT_DIR / "feature_importance_by_model.csv"

    all_results.to_csv(
        full_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Mean importance matrix
    # -------------------------------------------------------------------------

    importance_matrix = all_results.pivot(
        index="Feature",
        columns="Model",
        values="Importance_Mean",
    )

    importance_matrix = importance_matrix.fillna(0)

    importance_matrix_path = (
        REPORT_DIR / "feature_importance_matrix.csv"
    )

    importance_matrix.to_csv(
        importance_matrix_path
    )

    # -------------------------------------------------------------------------
    # Feature status matrix
    # -------------------------------------------------------------------------

    status_matrix = all_results.pivot(
        index="Feature",
        columns="Model",
        values="Feature_Status",
    )

    status_matrix_path = (
        REPORT_DIR / "feature_status_matrix.csv"
    )

    status_matrix.to_csv(
        status_matrix_path
    )

    # -------------------------------------------------------------------------
    # Binary influential-feature matrix
    # -------------------------------------------------------------------------

    influential = all_results.copy()

    influential["Influential"] = (
        influential["Feature_Status"]
        == "Meaningful Contribution"
    )

    influential_matrix = influential.pivot(
        index="Feature",
        columns="Model",
        values="Influential",
    )

    influential_matrix = influential_matrix.fillna(False)

    influential_matrix_path = (
        REPORT_DIR / "influential_feature_matrix.csv"
    )

    influential_matrix.to_csv(
        influential_matrix_path
    )

    # -------------------------------------------------------------------------
    # Model summary
    # -------------------------------------------------------------------------

    summary_rows = []

    for model_name in sorted(all_results["Model"].unique()):
        subset = all_results[
            all_results["Model"] == model_name
        ]

        meaningful_count = int(
            (
                subset["Feature_Status"]
                == "Meaningful Contribution"
            ).sum()
        )

        low_count = int(
            (
                subset["Feature_Status"]
                == "Low/Unclear Contribution"
            ).sum()
        )

        top_features = (
            subset.sort_values(
                "Absolute_Importance",
                ascending=False,
            )
            .head(5)["Feature"]
            .tolist()
        )

        summary_rows.append(
            {
                "Model": model_name,
                "Meaningful_Features": meaningful_count,
                "Low_or_Unclear_Features": low_count,
                "Top_5_Features": ", ".join(top_features),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    summary_path = (
        REPORT_DIR / "model_feature_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # JSON metadata
    # -------------------------------------------------------------------------

    metadata = {
        "dataset": "IBM Telco Customer Churn",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "permutation_repeats": N_REPEATS,
        "models": sorted(
            all_results["Model"].unique().tolist()
        ),
        "interpretation": (
            "Meaningful Contribution means the permutation "
            "importance lower bound (mean minus standard deviation) "
            "was positive in this experiment. "
            "Low/Unclear Contribution does not mean that the model "
            "cannot technically use the feature."
        ),
    }

    with (
        REPORT_DIR / "feature_analysis_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print("\nReports created:")
    for path in [
        full_path,
        importance_matrix_path,
        status_matrix_path,
        influential_matrix_path,
        summary_path,
    ]:
        print(f"  {path}")


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def create_visualizations(all_results: pd.DataFrame) -> None:
    """Create feature/model visualizations."""

    # -------------------------------------------------------------------------
    # Heatmap
    # -------------------------------------------------------------------------

    matrix = all_results.pivot(
        index="Feature",
        columns="Model",
        values="Importance_Mean",
    )

    matrix = matrix.fillna(0)

    # Sort by average absolute contribution.
    order = (
        matrix.abs()
        .mean(axis=1)
        .sort_values(ascending=False)
        .index
    )

    matrix = matrix.loc[order]

    plt.figure(
        figsize=(12, 9)
    )

    plt.imshow(
        matrix.values,
        aspect="auto",
    )

    plt.xticks(
        range(len(matrix.columns)),
        matrix.columns,
        rotation=30,
        ha="right",
    )

    plt.yticks(
        range(len(matrix.index)),
        matrix.index,
    )

    plt.xlabel("Model")
    plt.ylabel("Original Dataset Feature")
    plt.title(
        "Feature-Level Permutation Importance Across Five Models"
    )

    plt.colorbar(
        label="Mean ROC-AUC change"
    )

    plt.tight_layout()

    heatmap_path = (
        REPORT_DIR / "feature_model_heatmap.png"
    )

    plt.savefig(
        heatmap_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # -------------------------------------------------------------------------
    # Top-feature chart
    # -------------------------------------------------------------------------

    top_features = (
        all_results.groupby("Feature")[
            "Absolute_Importance"
        ]
        .mean()
        .sort_values(ascending=False)
        .head(12)
        .index
    )

    top_df = all_results[
        all_results["Feature"].isin(top_features)
    ].copy()

    pivot = top_df.pivot(
        index="Feature",
        columns="Model",
        values="Absolute_Importance",
    ).fillna(0)

    pivot = pivot.loc[
        pivot.mean(axis=1).sort_values(ascending=False).index
    ]

    plt.figure(
        figsize=(13, 8)
    )

    pivot.plot(
        kind="bar",
        ax=plt.gca(),
    )

    plt.title(
        "Top Features by Average Model Contribution"
    )
    plt.xlabel("Feature")
    plt.ylabel("Absolute Permutation Importance")
    plt.xticks(
        rotation=35,
        ha="right",
    )

    plt.tight_layout()

    bar_path = (
        REPORT_DIR / "top_feature_importance.png"
    )

    plt.savefig(
        bar_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print("\nVisualizations created:")
    print(f"  {heatmap_path}")
    print(f"  {bar_path}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("=" * 70)
    print("SELF-HEALING ML — FEATURE-LEVEL MODEL ANALYSIS")
    print("=" * 70)

    X, y = load_dataset()

    print(f"\nDataset shape: {X.shape}")
    print(f"Number of original features: {X.shape[1]}")
    print(f"Number of records: {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(f"\nTraining records: {len(X_train)}")
    print(f"Test records: {len(X_test)}")

    models = build_models()

    results = []

    for model_name, model in models.items():

        model_result = analyze_model(
            model_name=model_name,
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

        results.append(model_result)

    all_results = pd.concat(
        results,
        ignore_index=True,
    )

    create_reports(
        all_results
    )

    create_visualizations(
        all_results
    )

    print("\n" + "=" * 70)
    print("FEATURE ANALYSIS COMPLETED")
    print("=" * 70)

    print(
        f"\nAll reports are saved in:\n{REPORT_DIR}"
    )


if __name__ == "__main__":
    main()