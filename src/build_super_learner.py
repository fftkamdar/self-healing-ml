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
from sklearn.svm import SVC

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SEED = 42

ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = (
    ROOT
    / "reports"
    / "final_super_learner"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REFERENCE = {
    "Accuracy": 0.791341,
    "Precision": 0.639860,
    "Recall": 0.810160,
    "F1-score": 0.624742,
    "ROC-AUC": 0.835577,
}

V9 = {
    "Accuracy": 0.7864,
    "Precision": 0.5855,
    "Recall": 0.6684,
    "F1-score": 0.6242,
    "ROC-AUC": 0.8441,
}


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

    raise FileNotFoundError(
        "Telco Customer Churn CSV not found."
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

    if (
        "tenure" in df.columns
        and "MonthlyCharges" in df.columns
    ):
        df["TenureChargeInteraction"] = (
            df["tenure"]
            * df["MonthlyCharges"]
        )

    if "Contract" in df.columns:
        df["IsMonthToMonth"] = (
            df["Contract"]
            .astype(str)
            .eq("Month-to-month")
            .astype(int)
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

    # Additional interactions that are meaningful
    # for the churn problem.
    if (
        "MonthlyCharges" in df.columns
        and "Contract" in df.columns
    ):
        df["MonthlyCharge_MonthToMonth"] = (
            df["MonthlyCharges"]
            * df["Contract"]
            .astype(str)
            .eq("Month-to-month")
            .astype(int)
        )

    if (
        "tenure" in df.columns
        and "Contract" in df.columns
    ):
        df["Tenure_MonthToMonth"] = (
            df["tenure"]
            * df["Contract"]
            .astype(str)
            .eq("Month-to-month")
            .astype(int)
        )

    if (
        "MonthlyCharges" in df.columns
        and "InternetService" in df.columns
    ):
        df["Charge_InternetInteraction"] = (
            df["MonthlyCharges"]
            * df["InternetService"]
            .astype(str)
            .map(
                {
                    "DSL": 1.0,
                    "Fiber optic": 2.0,
                    "No": 0.0,
                }
            )
            .fillna(0.0)
        )

    return df


def prepare_data(df):
    if "Churn" not in df.columns:
        raise ValueError(
            "Churn column not found."
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
                numeric,
            ),
            (
                "cat",
                categorical_pipe,
                categorical,
            ),
        ]
    )


def build_base_models(y_train):
    positive = int(
        y_train.sum()
    )

    negative = int(
        len(y_train)
        - positive
    )

    scale_pos_weight = (
        negative / positive
        if positive
        else 1.0
    )

    models = {}

    models["lr"] = LogisticRegression(
        C=0.5,
        max_iter=4000,
        class_weight="balanced",
        random_state=SEED,
    )

    models["gb"] = GradientBoostingClassifier(
        n_estimators=220,
        learning_rate=0.04,
        max_depth=2,
        min_samples_leaf=8,
        subsample=0.90,
        random_state=SEED,
    )

    models["rf"] = RandomForestClassifier(
        n_estimators=450,
        max_depth=10,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=SEED,
        n_jobs=1,
    )

    models["svm"] = SVC(
        C=1.0,
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=SEED,
    )

    models["cat"] = CatBoostClassifier(
        iterations=500,
        depth=5,
        learning_rate=0.035,
        l2_leaf_reg=8.0,
        random_strength=1.0,
        bagging_temperature=0.5,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=SEED,
        verbose=False,
        thread_count=1,
        allow_writing_files=False,
    )

    models["lgb"] = LGBMClassifier(
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
        n_jobs=1,
        verbosity=-1,
    )

    models["xgb"] = XGBClassifier(
        n_estimators=400,
        max_depth=2,
        learning_rate=0.035,
        min_child_weight=5,
        subsample=0.90,
        colsample_bytree=0.85,
        gamma=0.05,
        reg_lambda=6.0,
        reg_alpha=0.10,
        scale_pos_weight=1.35,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=SEED,
        n_jobs=1,
    )

    return models


def fit_base(
    name,
    model,
    X_train,
    y_train,
):
    # CatBoost handles categoricals differently.
    if name == "cat":
        X_cat = X_train.copy()

        categorical = []

        for col in X_cat.columns:
            if not pd.api.types.is_numeric_dtype(
                X_cat[col]
            ):
                X_cat[col] = (
                    X_cat[col]
                    .fillna("Missing")
                    .astype(str)
                )
                categorical.append(col)

        model.fit(
            X_cat,
            y_train,
            cat_features=categorical,
        )

        return (
            model,
            categorical,
        )

    # LightGBM also gets native categorical columns.
    if name == "lgb":
        X_lgb = X_train.copy()

        categorical = []

        for col in X_lgb.columns:
            if not pd.api.types.is_numeric_dtype(
                X_lgb[col]
            ) and not pd.api.types.is_bool_dtype(
                X_lgb[col]
            ):
                X_lgb[col] = (
                    X_lgb[col]
                    .fillna("Missing")
                    .astype("category")
                )
                categorical.append(col)

        model.fit(
            X_lgb,
            y_train,
            categorical_feature=categorical,
        )

        return (
            model,
            categorical,
        )

    preprocessor = make_preprocessor(
        X_train
    )

    pipe = Pipeline(
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

    pipe.fit(
        X_train,
        y_train,
    )

    return (
        pipe,
        None,
    )


def predict_base(
    name,
    fitted,
    categorical_info,
    X,
):
    if name == "cat":
        X_cat = X.copy()

        for col in X_cat.columns:
            if (
                categorical_info
                and col in categorical_info
            ):
                X_cat[col] = (
                    X_cat[col]
                    .fillna("Missing")
                    .astype(str)
                )

        return (
            fitted.predict_proba(
                X_cat
            )[:, 1]
        )

    if name == "lgb":
        X_lgb = X.copy()

        for col in X_lgb.columns:
            if (
                categorical_info
                and col in categorical_info
            ):
                X_lgb[col] = (
                    X_lgb[col]
                    .fillna("Missing")
                    .astype("category")
                )

        return (
            fitted.predict_proba(
                X_lgb
            )[:, 1]
        )

    return (
        fitted.predict_proba(
            X
        )[:, 1]
    )


def calc_metrics(
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


def objective(metrics):
    ratios = [
        metrics[m] / REFERENCE[m]
        for m in REFERENCE
    ]

    return (
        int(
            all(
                ratio > 1.0
                for ratio in ratios
            )
        ),
        float(min(ratios)),
        float(np.mean(ratios)),
        float(metrics["F1-score"]),
        float(metrics["ROC-AUC"]),
    )


def make_meta_features(
    probability_matrix,
    X,
):
    result = pd.DataFrame(
        probability_matrix
    ).reset_index(
        drop=True
    )

    result.columns = [
        f"base_{i}"
        for i in range(
            result.shape[1]
        )
    ]

    # Add only stable numeric features
    numeric_candidates = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "AverageMonthlySpend",
        "ServiceCount",
        "FamilySupportCount",
        "IsMonthToMonth",
        "TenureChargeInteraction",
    ]

    for col in numeric_candidates:
        if col in X.columns:
            values = pd.to_numeric(
                X[col],
                errors="coerce",
            ).fillna(
                X[col]
                .median()
                if pd.api.types.is_numeric_dtype(
                    X[col]
                )
                else 0
            )

            result[
                f"feature_{col}"
            ] = values.reset_index(
                drop=True
            )

    return result


def build_meta_models():
    models = {}

    models[
        "Meta_Logistic"
    ] = Pipeline(
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
            (
                "model",
                LogisticRegression(
                    C=0.20,
                    max_iter=4000,
                    random_state=SEED,
                ),
            ),
        ]
    )

    models[
        "Meta_Logistic_Balanced"
    ] = Pipeline(
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
            (
                "model",
                LogisticRegression(
                    C=0.10,
                    class_weight="balanced",
                    max_iter=4000,
                    random_state=SEED,
                ),
            ),
        ]
    )

    models[
        "Meta_LightGBM"
    ] = LGBMClassifier(
        objective="binary",
        n_estimators=100,
        learning_rate=0.025,
        num_leaves=7,
        max_depth=3,
        min_child_samples=40,
        reg_alpha=0.50,
        reg_lambda=10.0,
        random_state=SEED,
        n_jobs=1,
        verbosity=-1,
    )

    return models


def main():
    start = time.time()

    print("=" * 100)
    print(
        "SH-DAS FINAL SUPER-LEARNER CHAMPION"
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

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,
            random_state=SEED,
        )
    )

    X_train = X_train.reset_index(
        drop=True
    )

    y_train = y_train.reset_index(
        drop=True
    )

    X_test = X_test.reset_index(
        drop=True
    )

    y_test = y_test.reset_index(
        drop=True
    )

    print(
        f"\nDevelopment: {len(X_train):,}"
    )

    print(
        f"Test:        {len(X_test):,}"
    )

    base_model_names = [
        "lr",
        "gb",
        "rf",
        "svm",
        "cat",
        "lgb",
        "xgb",
    ]

    folds = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    oof_matrix = np.zeros(
        (
            len(X_train),
            len(base_model_names),
        )
    )

    print(
        "\n5-FOLD OUT-OF-FOLD BASE MODEL TRAINING"
    )

    print(
        "-" * 100
    )

    for fold_no, (
        train_idx,
        valid_idx,
    ) in enumerate(
        folds.split(
            X_train,
            y_train,
        ),
        start=1,
    ):

        print(
            f"\nFOLD {fold_no}/5"
        )

        X_fold_train = (
            X_train.iloc[
                train_idx
            ]
        )

        y_fold_train = (
            y_train.iloc[
                train_idx
            ]
        )

        X_fold_valid = (
            X_train.iloc[
                valid_idx
            ]
        )

        models = build_base_models(
            y_fold_train
        )

        for model_position, name in enumerate(
            base_model_names
        ):

            print(
                f"  Training {name} ..."
            )

            fitted, categorical_info = (
                fit_base(
                    name,
                    models[name],
                    X_fold_train,
                    y_fold_train,
                )
            )

            probability = predict_base(
                name,
                fitted,
                categorical_info,
                X_fold_valid,
            )

            oof_matrix[
                valid_idx,
                model_position,
            ] = probability

    print(
        "\nOOF BASE MODELS COMPLETED"
    )

    oof_auc = {}

    for i, name in enumerate(
        base_model_names
    ):
        oof_auc[name] = float(
            roc_auc_score(
                y_train,
                oof_matrix[:, i],
            )
        )

        print(
            f"{name:<8} OOF AUC="
            f"{oof_auc[name]:.4f}"
        )

    meta_X = make_meta_features(
        oof_matrix,
        X_train,
    )

    # Outer CV over OOF predictions.
    outer_folds = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=77,
    )

    meta_models = build_meta_models()

    meta_results = []

    print(
        "\nMETA-MODEL OUTER VALIDATION"
    )

    print(
        "-" * 100
    )

    threshold_grid = np.arange(
        0.20,
        0.801,
        0.01,
    )

    for meta_name, meta_model in (
        meta_models.items()
    ):

        print(
            f"\nEvaluating {meta_name} ..."
        )

        outer_probability = (
            np.zeros(
                len(meta_X)
            )
        )

        for fold_no, (
            meta_train_idx,
            meta_valid_idx,
        ) in enumerate(
            outer_folds.split(
                meta_X,
                y_train,
            ),
            start=1,
        ):

            X_meta_train = (
                meta_X.iloc[
                    meta_train_idx
                ]
            )

            X_meta_valid = (
                meta_X.iloc[
                    meta_valid_idx
                ]
            )

            y_meta_train = (
                y_train.iloc[
                    meta_train_idx
                ]
            )

            fitted_meta = (
                build_meta_models()[
                    meta_name
                ]
            )

            if (
                meta_name
                == "Meta_LightGBM"
            ):
                fitted_meta.fit(
                    X_meta_train,
                    y_meta_train,
                )
            else:
                fitted_meta.fit(
                    X_meta_train,
                    y_meta_train,
                )

            outer_probability[
                meta_valid_idx
            ] = (
                fitted_meta
                .predict_proba(
                    X_meta_valid
                )[:, 1]
            )

            print(
                f"  Outer fold {fold_no}/5 complete"
            )

        best = None

        for threshold in threshold_grid:

            m = calc_metrics(
                y_train,
                outer_probability,
                float(threshold),
            )

            score = objective(
                m
            )

            record = {
                "MetaModel": meta_name,
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

        meta_results.append(
            best
        )

        print(
            f"  OOF Accuracy="
            f"{best['Accuracy']:.4f}"
        )

        print(
            f"  OOF Precision="
            f"{best['Precision']:.4f}"
        )

        print(
            f"  OOF Recall="
            f"{best['Recall']:.4f}"
        )

        print(
            f"  OOF F1="
            f"{best['F1-score']:.4f}"
        )

        print(
            f"  OOF AUC="
            f"{best['ROC-AUC']:.4f}"
        )

        print(
            f"  Threshold="
            f"{best['Threshold']:.2f}"
        )

    meta_results_df = (
        pd.DataFrame(
            meta_results
        )
        .sort_values(
            by=[
                "AllAbove",
                "MinRatio",
                "MeanRatio",
                "F1-score",
                "ROC-AUC",
            ],
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    selected_meta = (
        meta_results_df.iloc[0]
    )

    selected_meta_name = (
        selected_meta["MetaModel"]
    )

    selected_threshold = float(
        selected_meta["Threshold"]
    )

    print(
        "\n" + "=" * 100
    )

    print(
        "SELECTED META-MODEL"
    )

    print(
        "=" * 100
    )

    print(
        f"Model: "
        f"{selected_meta_name}"
    )

    print(
        f"Threshold: "
        f"{selected_threshold:.2f}"
    )

    # ---------------------------------------------------------
    # FINAL BASE MODELS ON FULL DEVELOPMENT DATA
    # ---------------------------------------------------------

    print(
        "\nTRAINING FINAL BASE MODELS "
        "ON FULL DEVELOPMENT DATA ..."
    )

    final_models = {}

    test_matrix = np.zeros(
        (
            len(X_test),
            len(base_model_names),
        )
    )

    full_base_models = (
        build_base_models(
            y_train
        )
    )

    for position, name in enumerate(
        base_model_names
    ):

        print(
            f"  Final {name} ..."
        )

        fitted, categorical_info = (
            fit_base(
                name,
                full_base_models[name],
                X_train,
                y_train,
            )
        )

        final_models[name] = (
            fitted,
            categorical_info,
        )

        test_matrix[
            :,
            position,
        ] = predict_base(
            name,
            fitted,
            categorical_info,
            X_test,
        )

    test_meta_X = make_meta_features(
        test_matrix,
        X_test,
    )

    # Final meta model trained on all OOF data.
    final_meta_models = (
        build_meta_models()
    )

    final_meta = (
        final_meta_models[
            selected_meta_name
        ]
    )

    final_meta.fit(
        meta_X,
        y_train,
    )

    final_probability = (
        final_meta.predict_proba(
            test_meta_X
        )[:, 1]
    )

    final_metrics = calc_metrics(
        y_test,
        final_probability,
        selected_threshold,
    )

    dominance = sum(
        final_metrics[m]
        > REFERENCE[m]
        for m in REFERENCE
    )

    v9_dominance = sum(
        final_metrics[m]
        > V9[m]
        for m in V9
    )

    print(
        "\n" + "=" * 100
    )

    print(
        "FINAL SUPER-LEARNER RESULT"
    )

    print(
        "=" * 100
    )

    for metric in REFERENCE:

        actual = final_metrics[
            metric
        ]

        reference = REFERENCE[
            metric
        ]

        symbol = (
            "PASS"
            if actual > reference
            else "MISS"
        )

        print(
            f"{metric:<12}: "
            f"{actual:.4f} "
            f"(canonical max: "
            f"{reference:.4f}) "
            f"{symbol}"
        )

    print(
        f"\nDOMINANCE COUNT: "
        f"{dominance}/5"
    )

    print(
        "BEATS ALL FIVE ON ALL FIVE METRICS: "
        + (
            "YES"
            if dominance == 5
            else "NO"
        )
    )

    print(
        f"SUPER-LEARNER vs V9 dominance: "
        f"{v9_dominance}/5"
    )

    comparison = pd.DataFrame(
        [
            {
                "Model": "SH-DAS Super-Learner",
                "Type": "Final proposed model",
                **final_metrics,
                "Threshold": selected_threshold,
                "CanonicalDominance": dominance,
            },
            {
                "Model": "SH-DAS V9",
                "Type": "Previous proposed model",
                **V9,
                "Threshold": 0.48,
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
        / "super_learner_final_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    oof_path = (
        REPORT_DIR
        / "super_learner_meta_results.csv"
    )

    meta_results_df.to_csv(
        oof_path,
        index=False,
    )

    base_auc_path = (
        REPORT_DIR
        / "base_model_oof_auc.json"
    )

    base_auc_path.write_text(
        json.dumps(
            oof_auc,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {
        "selected_meta_model": selected_meta_name,
        "threshold": selected_threshold,
        "base_models": base_model_names,
        "base_oof_auc": oof_auc,
        "test_metrics": final_metrics,
        "canonical_reference": REFERENCE,
        "v9_reference": V9,
        "dominance_count": int(
            dominance
        ),
        "beats_all_five": bool(
            dominance == 5
        ),
        "super_learner_vs_v9_dominance": int(
            v9_dominance
        ),
        "development_size": int(
            len(X_train)
        ),
        "test_size": int(
            len(X_test)
        ),
        "seed": SEED,
        "runtime_seconds": float(
            time.time()
            - start
        ),
    }

    summary_path = (
        REPORT_DIR
        / "super_learner_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "SH-DAS FINAL SUPER-LEARNER",
        "=" * 80,
        "",
        f"Selected meta-model: "
        f"{selected_meta_name}",
        f"Threshold: "
        f"{selected_threshold:.4f}",
        "",
        "FINAL TEST METRICS",
        "-" * 80,
    ]

    for metric in REFERENCE:
        report_lines.append(
            f"{metric}: "
            f"{final_metrics[metric]:.6f} "
            f"(canonical maximum: "
            f"{REFERENCE[metric]:.6f})"
        )

    report_lines.extend(
        [
            "",
            f"Dominance: {dominance}/5",
            (
                "Beats all five: "
                + (
                    "YES"
                    if dominance == 5
                    else "NO"
                )
            ),
            "",
            f"Super-Learner vs V9: "
            f"{v9_dominance}/5",
            "",
            "BASE MODELS",
            "-" * 80,
        ]
    )

    report_lines.extend(
        f"- {name}"
        for name in base_model_names
    )

    report_lines.extend(
        [
            "",
            f"Runtime: "
            f"{time.time() - start:.2f} seconds",
        ]
    )

    report_path = (
        REPORT_DIR
        / "super_learner_report.txt"
    )

    report_path.write_text(
        "\n".join(
            report_lines
        ),
        encoding="utf-8",
    )

    bundle_path = (
        REPORT_DIR
        / "sh_das_super_learner_bundle.joblib"
    )

    joblib.dump(
        {
            "base_models": final_models,
            "meta_model": final_meta,
            "base_model_names": base_model_names,
            "threshold": selected_threshold,
            "selected_meta_model": selected_meta_name,
            "seed": SEED,
        },
        bundle_path,
    )

    print(
        "\nSAVED"
    )

    print("-" * 100)
    print(comparison_path)
    print(oof_path)
    print(base_auc_path)
    print(summary_path)
    print(report_path)
    print(bundle_path)

    print(
        f"\nRuntime: "
        f"{time.time() - start:.2f} seconds"
    )


if __name__ == "__main__":
    main()
