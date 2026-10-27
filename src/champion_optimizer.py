from pathlib import Path
import json
import time
import warnings
from itertools import product

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
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
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SEED = 42
FOLDS = 5

ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = (
    ROOT
    / "reports"
    / "champion_optimizer"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# ------------------------------------------------------------
# FROZEN CANONICAL BENCHMARK
# ------------------------------------------------------------

REFERENCE = {
    "Accuracy": 0.791341,
    "Precision": 0.639860,
    "Recall": 0.810160,
    "F1-score": 0.624742,
    "ROC-AUC": 0.835577,
}

# Existing strongest proposed result observed so far.
SUPER_REFERENCE = {
    "Accuracy": 0.7779,
    "Precision": 0.5616,
    "Recall": 0.7433,
    "F1-score": 0.6398,
    "ROC-AUC": 0.8461,
}


def find_dataset():

    candidates = [
        ROOT
        / "data"
        / "raw"
        / "Telco-Customer-Churn.csv",
        ROOT
        / "data"
        / "Telco-Customer-Churn.csv",
        ROOT
        / "Telco-Customer-Churn.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    for path in ROOT.rglob("*.csv"):
        name = path.name.lower()

        if "telco" in name or "churn" in name:
            return path

    raise FileNotFoundError(
        "Telco-Customer-Churn.csv was not found."
    )


def engineer_features(df):

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

    # --------------------------------------------------------
    # Core numerical interactions
    # --------------------------------------------------------

    if (
        "tenure" in df.columns
        and "MonthlyCharges" in df.columns
    ):

        df["TenureChargeInteraction"] = (
            df["tenure"]
            * df["MonthlyCharges"]
        )

        df["ChargePerTenureMonth"] = (
            df["MonthlyCharges"]
            / (
                df["tenure"]
                + 1.0
            )
        )

    if (
        "tenure" in df.columns
        and "TotalCharges" in df.columns
    ):

        df["AverageMonthlySpend"] = (
            df["TotalCharges"]
            / df["tenure"].replace(
                0,
                np.nan,
            )
        )

        df["TotalChargePerTenure"] = (
            df["TotalCharges"]
            / (
                df["tenure"]
                + 1.0
            )
        )

    # --------------------------------------------------------
    # Contract indicators
    # --------------------------------------------------------

    if "Contract" in df.columns:

        df["IsMonthToMonth"] = (
            df["Contract"]
            .astype(str)
            .eq("Month-to-month")
            .astype(int)
        )

        df["IsLongTermContract"] = (
            df["Contract"]
            .astype(str)
            .isin(
                [
                    "One year",
                    "Two year",
                ]
            )
            .astype(int)
        )

    # --------------------------------------------------------
    # Service counts
    # --------------------------------------------------------

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

    existing_services = [
        c
        for c in service_columns
        if c in df.columns
    ]

    if existing_services:

        df["ServiceCount"] = 0

        for col in existing_services:

            df["ServiceCount"] += (
                ~df[col]
                .astype(str)
                .isin(
                    [
                        "No",
                        "No internet service",
                        "nan",
                    ]
                )
            ).astype(int)

    support_columns = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
    ]

    existing_support = [
        c
        for c in support_columns
        if c in df.columns
    ]

    if existing_support:

        df["SupportServiceCount"] = 0

        for col in existing_support:

            df["SupportServiceCount"] += (
                df[col]
                .astype(str)
                .eq("Yes")
            ).astype(int)

    # --------------------------------------------------------
    # Family
    # --------------------------------------------------------

    if (
        "Partner" in df.columns
        and "Dependents" in df.columns
    ):

        df["FamilySupportCount"] = (
            df["Partner"]
            .astype(str)
            .eq("Yes")
            .astype(int)
            +
            df["Dependents"]
            .astype(str)
            .eq("Yes")
            .astype(int)
        )

    # --------------------------------------------------------
    # Payment / Internet indicators
    # --------------------------------------------------------

    if "PaymentMethod" in df.columns:

        df["IsElectronicCheck"] = (
            df["PaymentMethod"]
            .astype(str)
            .eq("Electronic check")
            .astype(int)
        )

    if "InternetService" in df.columns:

        df["IsFiberOptic"] = (
            df["InternetService"]
            .astype(str)
            .eq("Fiber optic")
            .astype(int)
        )

    # --------------------------------------------------------
    # Stable interaction categories
    # --------------------------------------------------------

    def interaction(
        new_column,
        first,
        second,
    ):

        if (
            first in df.columns
            and second in df.columns
        ):

            df[new_column] = (
                df[first]
                .astype(str)
                + "__"
                + df[second]
                .astype(str)
            )

    interaction(
        "Contract_Internet",
        "Contract",
        "InternetService",
    )

    interaction(
        "Contract_Payment",
        "Contract",
        "PaymentMethod",
    )

    interaction(
        "Internet_Support",
        "InternetService",
        "TechSupport",
    )

    interaction(
        "Internet_Security",
        "InternetService",
        "OnlineSecurity",
    )

    return df


def prepare_data(df):

    if "Churn" not in df.columns:
        raise ValueError(
            "Churn target column not found."
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
            "Unexpected values in Churn."
        )

    X = df.drop(
        columns=["Churn"]
    )

    return X, y.astype(int)


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
        ],
        remainder="drop",
    )


def build_models():

    return {

        # Strong recall component
        "LR": Pipeline(
            [
                (
                    "preprocessor",
                    None,
                ),
                (
                    "model",
                    LogisticRegression(
                        C=0.40,
                        max_iter=4000,
                        class_weight="balanced",
                        random_state=SEED,
                    ),
                ),
            ]
        ),

        # Strong precision / accuracy component
        "GB": Pipeline(
            [
                (
                    "preprocessor",
                    None,
                ),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=220,
                        learning_rate=0.035,
                        max_depth=2,
                        min_samples_leaf=10,
                        subsample=0.90,
                        random_state=SEED,
                    ),
                ),
            ]
        ),

        # Nonlinear component 1
        "XGB": Pipeline(
            [
                (
                    "preprocessor",
                    None,
                ),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=450,
                        max_depth=2,
                        learning_rate=0.030,
                        min_child_weight=6,
                        subsample=0.90,
                        colsample_bytree=0.85,
                        gamma=0.05,
                        reg_lambda=7.0,
                        reg_alpha=0.10,
                        scale_pos_weight=1.20,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        tree_method="hist",
                        random_state=SEED,
                        n_jobs=2,
                    ),
                ),
            ]
        ),

        # Nonlinear component 2
        "CAT": Pipeline(
            [
                (
                    "preprocessor",
                    None,
                ),
                (
                    "model",
                    CatBoostClassifier(
                        iterations=450,
                        depth=5,
                        learning_rate=0.035,
                        l2_leaf_reg=9.0,
                        random_strength=1.0,
                        bagging_temperature=0.5,
                        loss_function="Logloss",
                        eval_metric="AUC",
                        random_seed=SEED,
                        verbose=False,
                        thread_count=2,
                        allow_writing_files=False,
                        class_weights=[1.0, 1.15],
                    ),
                ),
            ]
        ),

        # Nonlinear component 3
        "LGB": Pipeline(
            [
                (
                    "preprocessor",
                    None,
                ),
                (
                    "model",
                    LGBMClassifier(
                        objective="binary",
                        n_estimators=450,
                        learning_rate=0.025,
                        num_leaves=12,
                        max_depth=4,
                        min_child_samples=30,
                        subsample=0.90,
                        colsample_bytree=0.90,
                        reg_alpha=0.10,
                        reg_lambda=6.0,
                        class_weight=None,
                        random_state=SEED,
                        n_jobs=2,
                        verbosity=-1,
                    ),
                ),
            ]
        ),
    }


def fit_pipeline(model_name, X_train, y_train):

    models = build_models()

    model = models[model_name]

    if model_name == "CAT":

        X_local = X_train.copy()

        categorical_columns = []

        for col in X_local.columns:

            if (
                not pd.api.types
                .is_numeric_dtype(
                    X_local[col]
                )
                and
                not pd.api.types
                .is_bool_dtype(
                    X_local[col]
                )
            ):

                X_local[col] = (
                    X_local[col]
                    .fillna("Missing")
                    .astype(str)
                )

                categorical_columns.append(col)

        model.set_params(
            preprocessor=None
        )

        model.named_steps[
            "model"
        ].fit(
            X_local,
            y_train,
            cat_features=categorical_columns,
        )

        return model, categorical_columns

    if model_name == "LGB":

        X_local = X_train.copy()

        categorical_columns = []

        for col in X_local.columns:

            if (
                not pd.api.types
                .is_numeric_dtype(
                    X_local[col]
                )
                and
                not pd.api.types
                .is_bool_dtype(
                    X_local[col]
                )
            ):

                X_local[col] = (
                    X_local[col]
                    .fillna("Missing")
                    .astype("category")
                )

                categorical_columns.append(col)

        model.set_params(
            preprocessor=None
        )

        model.named_steps[
            "model"
        ].fit(
            X_local,
            y_train,
            categorical_feature=categorical_columns,
        )

        return model, categorical_columns

    preprocessor = make_preprocessor(
        X_train
    )

    model.steps[0] = (
        "preprocessor",
        preprocessor
    )

    model.fit(
        X_train,
        y_train,
    )

    return model, None


def predict_pipeline(
    model_name,
    fitted_model,
    categorical_columns,
    X,
):

    if model_name == "CAT":

        X_local = X.copy()

        for col in categorical_columns:

            X_local[col] = (
                X_local[col]
                .fillna("Missing")
                .astype(str)
            )

        return (
            fitted_model
            .named_steps["model"]
            .predict_proba(
                X_local
            )[:, 1]
        )

    if model_name == "LGB":

        X_local = X.copy()

        for col in categorical_columns:

            X_local[col] = (
                X_local[col]
                .fillna("Missing")
                .astype("category")
            )

        return (
            fitted_model
            .named_steps["model"]
            .predict_proba(
                X_local
            )[:, 1]
        )

    return (
        fitted_model
        .predict_proba(X)[:, 1]
    )


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


def metric_floor(metrics):

    ratios = [
        metrics[m]
        / REFERENCE[m]
        for m in REFERENCE
    ]

    return float(
        min(ratios)
    )


def metric_mean_ratio(metrics):

    ratios = [
        metrics[m]
        / REFERENCE[m]
        for m in REFERENCE
    ]

    return float(
        np.mean(ratios)
    )


def main():

    start = time.time()

    print("=" * 100)
    print(
        "SH-DAS CHAMPION OPTIMIZER"
    )
    print("=" * 100)

    dataset = find_dataset()

    print(
        f"\nDataset: {dataset}"
    )

    df = pd.read_csv(
        dataset
    )

    df = engineer_features(
        df
    )

    X, y = prepare_data(
        df
    )

    # EXACT CANONICAL HOLDOUT
    X_development, X_test, y_development, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,
            random_state=SEED,
        )
    )

    X_development = (
        X_development
        .reset_index(drop=True)
    )

    y_development = (
        y_development
        .reset_index(drop=True)
    )

    X_test = (
        X_test
        .reset_index(drop=True)
    )

    y_test = (
        y_test
        .reset_index(drop=True)
    )

    print(
        f"\nDevelopment: "
        f"{len(X_development):,}"
    )

    print(
        f"Test: "
        f"{len(X_test):,}"
    )

    model_names = [
        "LR",
        "GB",
        "XGB",
        "CAT",
        "LGB",
    ]

    # --------------------------------------------------------
    # OOF PREDICTIONS
    # --------------------------------------------------------

    folds = StratifiedKFold(
        n_splits=FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    oof = {
        name: np.zeros(
            len(X_development)
        )
        for name in model_names
    }

    # Store each fold's predictions.
    fold_probabilities = {
        name: []
        for name in model_names
    }

    fold_indices = []

    print(
        "\n5-FOLD CROSS-VALIDATION"
    )

    print(
        "-" * 100
    )

    for fold_no, (
        train_idx,
        valid_idx,
    ) in enumerate(
        folds.split(
            X_development,
            y_development,
        ),
        start=1,
    ):

        print(
            f"\nFOLD {fold_no}/{FOLDS}"
        )

        fold_indices.append(
            valid_idx
        )

        X_fold_train = (
            X_development.iloc[
                train_idx
            ]
        )

        X_fold_valid = (
            X_development.iloc[
                valid_idx
            ]
        )

        y_fold_train = (
            y_development.iloc[
                train_idx
            ]
        )

        y_fold_valid = (
            y_development.iloc[
                valid_idx
            ]
        )

        for name in model_names:

            print(
                f"  Training {name} ..."
            )

            fitted, cat_info = (
                fit_pipeline(
                    name,
                    X_fold_train,
                    y_fold_train,
                )
            )

            probability = (
                predict_pipeline(
                    name,
                    fitted,
                    cat_info,
                    X_fold_valid,
                )
            )

            oof[name][
                valid_idx
            ] = probability

            fold_probabilities[
                name
            ].append(
                (
                    valid_idx,
                    probability,
                )
            )

    print(
        "\nOOF MODEL QUALITY"
    )

    print(
        "-" * 100
    )

    for name in model_names:

        auc = roc_auc_score(
            y_development,
            oof[name],
        )

        print(
            f"{name:<10} "
            f"OOF AUC={auc:.4f}"
        )

    # --------------------------------------------------------
    # STABLE WEIGHT SEARCH
    # --------------------------------------------------------

    print(
        "\nSEARCHING STABLE ENSEMBLE WEIGHTS"
    )

    print(
        "-" * 100
    )

    # Only 5 models.
    # Weight step is 0.05.
    #
    # Number of valid combinations:
    # approximately 10,000.
    #
    # Search is performed only on
    # already-generated OOF arrays,
    # so this stage is extremely fast.

    # Controlled search grid: enough coverage without
    # hundreds of thousands of metric evaluations.
    weight_values = np.arange(
        0.0,
        1.001,
        0.10,
    )

    thresholds = np.arange(
        0.25,
        0.751,
        0.02,
    )

    best = None

    model_order = model_names

    search_counter = 0

    for w_lr in weight_values:

        for w_gb in weight_values:

            for w_xgb in weight_values:

                for w_cat in weight_values:

                    w_lgb = (
                        1.0
                        - w_lr
                        - w_gb
                        - w_xgb
                        - w_cat
                    )

                    if w_lgb < -1e-9:
                        continue

                    weights = np.array(
                        [
                            w_lr,
                            w_gb,
                            w_xgb,
                            w_cat,
                            w_lgb,
                        ]
                    )

                    if (
                        abs(
                            weights.sum()
                            - 1.0
                        )
                        > 1e-8
                    ):
                        continue

                    # Avoid useless single-model
                    # extreme points during search.
                    if np.count_nonzero(
                        weights > 0.0
                    ) < 2:
                        continue

                    probability = (
                        weights[0]
                        * oof["LR"]
                        +
                        weights[1]
                        * oof["GB"]
                        +
                        weights[2]
                        * oof["XGB"]
                        +
                        weights[3]
                        * oof["CAT"]
                        +
                        weights[4]
                        * oof["LGB"]
                    )

                    # Evaluate thresholds.
                    for threshold in thresholds:

                        metrics = calculate_metrics(
                            y_development,
                            probability,
                            float(threshold),
                        )

                        floor = metric_floor(
                            metrics
                        )

                        mean_ratio = (
                            metric_mean_ratio(
                                metrics
                            )
                        )

                        # We want a robust floor.
                        score = (
                            floor,
                            mean_ratio,
                            metrics["F1-score"],
                            metrics["ROC-AUC"],
                        )

                        if best is None:
                            best = {
                                "weights": weights.copy(),
                                "threshold": float(
                                    threshold
                                ),
                                "metrics": metrics,
                                "score": score,
                            }

                        elif score > best["score"]:
                            best = {
                                "weights": weights.copy(),
                                "threshold": float(
                                    threshold
                                ),
                                "metrics": metrics,
                                "score": score,
                            }

                    search_counter += 1

    print(
        f"\nWeight combinations evaluated: "
        f"{search_counter:,}"
    )

    print(
        "\nOOF SELECTED CHAMPION"
    )

    print(
        "-" * 100
    )

    for name, weight in zip(
        model_order,
        best["weights"],
    ):
        print(
            f"{name:<10}: "
            f"{weight:.2f}"
        )

    print(
        f"Threshold: "
        f"{best['threshold']:.2f}"
    )

    for metric in REFERENCE:

        print(
            f"{metric:<12}: "
            f"{best['metrics'][metric]:.4f}"
        )

    print(
        f"\nOOF minimum normalized score: "
        f"{best['score'][0]:.4f}"
    )

    # --------------------------------------------------------
    # FINAL FIT
    # --------------------------------------------------------

    print(
        "\nFINAL MODEL FIT ON FULL DEVELOPMENT DATA"
    )

    print(
        "-" * 100
    )

    final_models = {}

    test_probability_matrix = []

    for name in model_names:

        print(
            f"Training final {name} ..."
        )

        fitted, cat_info = (
            fit_pipeline(
                name,
                X_development,
                y_development,
            )
        )

        test_probability = (
            predict_pipeline(
                name,
                fitted,
                cat_info,
                X_test,
            )
        )

        final_models[name] = (
            fitted,
            cat_info,
        )

        test_probability_matrix.append(
            test_probability
        )

    test_probability_matrix = (
        np.column_stack(
            test_probability_matrix
        )
    )

    final_probability = (
        test_probability_matrix
        @ best["weights"]
    )

    final_threshold = float(
        best["threshold"]
    )

    final_metrics = calculate_metrics(
        y_test,
        final_probability,
        final_threshold,
    )

    canonical_dominance = sum(
        final_metrics[m]
        > REFERENCE[m]
        for m in REFERENCE
    )

    super_dominance = sum(
        final_metrics[m]
        > SUPER_REFERENCE[m]
        for m in SUPER_REFERENCE
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print(
        "\n" + "=" * 100
    )

    print(
        "FINAL SH-DAS CHAMPION"
    )

    print(
        "=" * 100
    )

    print(
        "\nSelected ensemble:"
    )

    for name, weight in zip(
        model_order,
        best["weights"],
    ):

        print(
            f"  {name:<8}: "
            f"{weight:.2f}"
        )

    print(
        f"\nThreshold: "
        f"{final_threshold:.2f}"
    )

    print(
        "\nFINAL TEST METRICS"
    )

    print(
        "-" * 100
    )

    for metric in REFERENCE:

        actual = final_metrics[
            metric
        ]

        target = REFERENCE[
            metric
        ]

        difference = (
            actual - target
        )

        symbol = (
            "PASS"
            if difference > 0
            else "MISS"
        )

        print(
            f"{metric:<12}: "
            f"{actual:.4f} "
            f"(target={target:.4f}) "
            f"{difference:+.4f} "
            f"{symbol}"
        )

    print(
        f"\nCANONICAL DOMINANCE: "
        f"{canonical_dominance}/5"
    )

    print(
        "BEATS ALL FIVE ON ALL FIVE METRICS: "
        + (
            "YES"
            if canonical_dominance == 5
            else "NO"
        )
    )

    print(
        f"CHAMPION vs SUPER-LEARNER: "
        f"{super_dominance}/5"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    comparison = pd.DataFrame(
        [
            {
                "Model": "SH-DAS Champion",
                "Type": "Optimized fixed ensemble",
                **final_metrics,
                "Threshold": final_threshold,
                "CanonicalDominance": canonical_dominance,
            },
            {
                "Model": "Super-Learner",
                "Type": "Previous proposed model",
                **SUPER_REFERENCE,
                "Threshold": 0.58,
                "CanonicalDominance": np.nan,
            },
            {
                "Model": "Canonical Maximum",
                "Type": "Frozen five-model benchmark",
                **REFERENCE,
                "Threshold": np.nan,
                "CanonicalDominance": np.nan,
            },
        ]
    )

    comparison_path = (
        REPORT_DIR
        / "champion_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    weights_path = (
        REPORT_DIR
        / "champion_weights.json"
    )

    weights_path.write_text(
        json.dumps(
            {
                name: float(weight)
                for name, weight in zip(
                    model_order,
                    best["weights"],
                )
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {
        "selected_models": model_order,
        "selected_weights": {
            name: float(weight)
            for name, weight in zip(
                model_order,
                best["weights"],
            )
        },
        "threshold": final_threshold,
        "final_metrics": final_metrics,
        "canonical_reference": REFERENCE,
        "super_learner_reference": SUPER_REFERENCE,
        "canonical_dominance": int(
            canonical_dominance
        ),
        "beats_all_five": bool(
            canonical_dominance == 5
        ),
        "champion_vs_super_learner": int(
            super_dominance
        ),
        "development_size": int(
            len(X_development)
        ),
        "test_size": int(
            len(X_test)
        ),
        "folds": FOLDS,
        "feature_count": int(
            len(X.columns)
        ),
        "runtime_seconds": float(
            time.time()
            - start
        ),
    }

    summary_path = (
        REPORT_DIR
        / "champion_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "SH-DAS CHAMPION",
        "=" * 80,
        "",
        "SELECTED ENSEMBLE",
        "-" * 80,
    ]

    for name, weight in zip(
        model_order,
        best["weights"],
    ):
        report_lines.append(
            f"{name}: {weight:.4f}"
        )

    report_lines.extend(
        [
            "",
            f"Threshold: "
            f"{final_threshold:.4f}",
            "",
            "FINAL TEST METRICS",
            "-" * 80,
        ]
    )

    for metric in REFERENCE:

        report_lines.append(
            f"{metric}: "
            f"{final_metrics[metric]:.6f} "
            f"(canonical target: "
            f"{REFERENCE[metric]:.6f})"
        )

    report_lines.extend(
        [
            "",
            f"Canonical dominance: "
            f"{canonical_dominance}/5",
            (
                "Beats all five: "
                + (
                    "YES"
                    if canonical_dominance == 5
                    else "NO"
                )
            ),
            "",
            f"Champion vs Super-Learner: "
            f"{super_dominance}/5",
            "",
            f"Feature count: "
            f"{len(X.columns)}",
            f"Runtime: "
            f"{time.time() - start:.2f} seconds",
        ]
    )

    report_path = (
        REPORT_DIR
        / "champion_report.txt"
    )

    report_path.write_text(
        "\n".join(
            report_lines
        ),
        encoding="utf-8",
    )

    bundle_path = (
        REPORT_DIR
        / "sh_das_champion_bundle.joblib"
    )

    joblib.dump(
        {
            "models": final_models,
            "weights": {
                name: float(weight)
                for name, weight in zip(
                    model_order,
                    best["weights"],
                )
            },
            "threshold": final_threshold,
            "seed": SEED,
        },
        bundle_path,
    )

    print(
        "\nSAVED"
    )

    print(
        "-" * 100
    )

    print(comparison_path)
    print(weights_path)
    print(summary_path)
    print(report_path)
    print(bundle_path)

    print(
        f"\nRuntime: "
        f"{time.time() - start:.2f} seconds"
    )


if __name__ == "__main__":
    main()


