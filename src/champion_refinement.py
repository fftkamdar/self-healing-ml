from pathlib import Path
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
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
from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SEED = 42
FOLDS = 4
TARGET = "Churn"

ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = (
    ROOT
    / "reports"
    / "champion_refinement"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# FROZEN CANONICAL BENCHMARK
REFERENCE = {
    "Accuracy": 0.791341,
    "Precision": 0.639860,
    "Recall": 0.810160,
    "F1-score": 0.624742,
    "ROC-AUC": 0.835577,
}

# CURRENT STRONGEST PROPOSED RESULT
SUPER_LEARNER_REFERENCE = {
    "Accuracy": 0.7779,
    "Precision": 0.5616,
    "Recall": 0.7433,
    "F1-score": 0.6398,
    "ROC-AUC": 0.8461,
}

CATEGORICAL_BASE = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


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

    # -----------------------------
    # Numeric features
    # -----------------------------

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

    if (
        "tenure" in df.columns
        and "TotalCharges" in df.columns
    ):
        df["TotalChargePerTenure"] = (
            df["TotalCharges"]
            / (
                df["tenure"]
                + 1.0
            )
        )

    # -----------------------------
    # Service features
    # -----------------------------

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

    security_support = [
        "OnlineSecurity",
        "TechSupport",
        "DeviceProtection",
    ]

    security_existing = [
        c
        for c in security_support
        if c in df.columns
    ]

    if security_existing:
        df["SecuritySupportCount"] = 0

        for col in security_existing:
            df["SecuritySupportCount"] += (
                df[col]
                .astype(str)
                .eq("Yes")
            ).astype(int)

    streaming_columns = [
        "StreamingTV",
        "StreamingMovies",
    ]

    streaming_existing = [
        c
        for c in streaming_columns
        if c in df.columns
    ]

    if streaming_existing:
        df["StreamingCount"] = 0

        for col in streaming_existing:
            df["StreamingCount"] += (
                df[col]
                .astype(str)
                .eq("Yes")
            ).astype(int)

    if "Contract" in df.columns:
        df["IsMonthToMonth"] = (
            df["Contract"]
            .astype(str)
            .eq("Month-to-month")
            .astype(int)
        )

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

    # -----------------------------
    # Fixed numeric bands
    # -----------------------------

    if "tenure" in df.columns:
        df["TenureBand"] = pd.cut(
            df["tenure"],
            bins=[
                -1,
                6,
                12,
                24,
                36,
                60,
                72,
            ],
            labels=[
                "T0_6",
                "T7_12",
                "T13_24",
                "T25_36",
                "T37_60",
                "T61_72",
            ],
        ).astype(str)

    if "MonthlyCharges" in df.columns:
        df["MonthlyChargeBand"] = pd.cut(
            df["MonthlyCharges"],
            bins=[
                -1,
                30,
                50,
                70,
                90,
                120,
            ],
            labels=[
                "C0_30",
                "C31_50",
                "C51_70",
                "C71_90",
                "C91_120",
            ],
        ).astype(str)

    # -----------------------------
    # Interaction categories
    # -----------------------------

    def safe_interaction(
        new_col,
        first,
        second,
    ):
        if (
            first in df.columns
            and second in df.columns
        ):
            df[new_col] = (
                df[first]
                .astype(str)
                + "_"
                + df[second]
                .astype(str)
            )

    safe_interaction(
        "Contract_Internet",
        "Contract",
        "InternetService",
    )

    safe_interaction(
        "Contract_Payment",
        "Contract",
        "PaymentMethod",
    )

    safe_interaction(
        "Contract_TenureBand",
        "Contract",
        "TenureBand",
    )

    safe_interaction(
        "Internet_TechSupport",
        "InternetService",
        "TechSupport",
    )

    safe_interaction(
        "Internet_Security",
        "InternetService",
        "OnlineSecurity",
    )

    safe_interaction(
        "Contract_ChargeBand",
        "Contract",
        "MonthlyChargeBand",
    )

    return df


def prepare_target(df):
    if TARGET not in df.columns:
        raise ValueError(
            "Churn column not found."
        )

    y = (
        df[TARGET]
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
            "Unexpected Churn values found."
        )

    X = df.drop(
        columns=[TARGET]
    )

    return X, y.astype(int)


def get_categorical_columns(X):
    return [
        c
        for c in X.columns
        if (
            not pd.api.types.is_numeric_dtype(
                X[c]
            )
            and
            not pd.api.types.is_bool_dtype(
                X[c]
            )
        )
    ]


def fit_target_encoder(
    X_train,
    y_train,
    categorical_columns,
    smoothing=20.0,
):
    global_mean = float(
        y_train.mean()
    )

    target_maps = {}
    frequency_maps = {}

    for col in categorical_columns:

        temp = pd.DataFrame(
            {
                "feature": X_train[col]
                .astype(str)
                .fillna("Missing"),
                "target": y_train
                .reset_index(
                    drop=True
                ),
            }
        )

        grouped = (
            temp.groupby(
                "feature"
            )["target"]
            .agg(
                [
                    "sum",
                    "count",
                ]
            )
        )

        smoothed = (
            grouped["sum"]
            + smoothing * global_mean
        ) / (
            grouped["count"]
            + smoothing
        )

        target_maps[col] = (
            smoothed.to_dict()
        )

        frequencies = (
            X_train[col]
            .astype(str)
            .fillna("Missing")
            .value_counts(
                normalize=True
            )
            .to_dict()
        )

        frequency_maps[col] = frequencies

    return (
        target_maps,
        frequency_maps,
        global_mean,
    )


def transform_target_encoded(
    X,
    encoder,
):
    target_maps, frequency_maps, global_mean = (
        encoder
    )

    result = pd.DataFrame(
        index=X.index
    )

    categorical_columns = list(
        target_maps.keys()
    )

    # Numeric columns
    for col in X.columns:

        if col in categorical_columns:
            continue

        values = pd.to_numeric(
            X[col],
            errors="coerce",
        )

        result[col] = values

    # Target and frequency encoding
    for col in categorical_columns:

        values = (
            X[col]
            .astype(str)
            .fillna("Missing")
        )

        result[
            f"{col}__TE"
        ] = (
            values
            .map(
                target_maps[col]
            )
            .fillna(
                global_mean
            )
            .astype(float)
        )

        result[
            f"{col}__FREQ"
        ] = (
            values
            .map(
                frequency_maps[col]
            )
            .fillna(0.0)
            .astype(float)
        )

    return result


def fit_transform_fold(
    X_train,
    y_train,
    X_other,
):
    categorical_columns = (
        get_categorical_columns(
            X_train
        )
    )

    encoder = fit_target_encoder(
        X_train,
        y_train,
        categorical_columns,
        smoothing=20.0,
    )

    train_encoded = (
        transform_target_encoded(
            X_train,
            encoder,
        )
    )

    other_encoded = (
        transform_target_encoded(
            X_other,
            encoder,
        )
    )

    numeric_imputer = (
        SimpleImputer(
            strategy="median"
        )
    )

    train_encoded = pd.DataFrame(
        numeric_imputer.fit_transform(
            train_encoded
        ),
        columns=train_encoded.columns,
        index=train_encoded.index,
    )

    other_encoded = pd.DataFrame(
        numeric_imputer.transform(
            other_encoded
        ),
        columns=train_encoded.columns,
        index=other_encoded.index,
    )

    return (
        train_encoded,
        other_encoded,
        encoder,
        numeric_imputer,
    )


def build_models():
    return {
        "LR_Standard": Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    LogisticRegression(
                        C=0.35,
                        max_iter=4000,
                        random_state=SEED,
                    ),
                ),
            ]
        ),

        "LR_Balanced": Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    LogisticRegression(
                        C=0.50,
                        class_weight="balanced",
                        max_iter=4000,
                        random_state=SEED,
                    ),
                ),
            ]
        ),

        "HistGradientBoosting":
            HistGradientBoostingClassifier(
                max_iter=250,
                learning_rate=0.035,
                max_leaf_nodes=15,
                min_samples_leaf=25,
                l2_regularization=2.0,
                random_state=SEED,
            ),

        "GradientBoosting":
            GradientBoostingClassifier(
                n_estimators=220,
                learning_rate=0.035,
                max_depth=2,
                min_samples_leaf=10,
                subsample=0.90,
                random_state=SEED,
            ),

        "LightGBM":
            LGBMClassifier(
                objective="binary",
                n_estimators=500,
                learning_rate=0.025,
                num_leaves=12,
                max_depth=4,
                min_child_samples=30,
                subsample=0.90,
                colsample_bytree=0.90,
                reg_alpha=0.10,
                reg_lambda=6.0,
                random_state=SEED,
                n_jobs=2,
                verbosity=-1,
            ),

        "XGBoost":
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

        "RandomForest":
            RandomForestClassifier(
                n_estimators=400,
                max_depth=10,
                min_samples_leaf=3,
                max_features="sqrt",
                class_weight="balanced",
                random_state=SEED,
                n_jobs=2,
            ),
    }


def fit_models_for_fold(
    X_fold_train_encoded,
    y_fold_train,
):
    models = build_models()

    fitted = {}

    for name, model in models.items():

        model.fit(
            X_fold_train_encoded,
            y_fold_train,
        )

        fitted[name] = model

    return fitted


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
):
    frame = pd.DataFrame(
        probability_matrix
    )

    frame.columns = [
        f"p_{i}"
        for i in range(
            frame.shape[1]
        )
    ]

    # Logit probabilities
    clipped = np.clip(
        probability_matrix,
        1e-5,
        1 - 1e-5,
    )

    logits = np.log(
        clipped
        /
        (
            1.0
            - clipped
        )
    )

    logit_frame = pd.DataFrame(
        logits,
        columns=[
            f"logit_{i}"
            for i in range(
                logits.shape[1]
            )
        ],
    )

    frame = pd.concat(
        [
            frame.reset_index(
                drop=True
            ),
            logit_frame.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    frame["prob_mean"] = (
        probability_matrix.mean(
            axis=1
        )
    )

    frame["prob_std"] = (
        probability_matrix.std(
            axis=1
        )
    )

    frame["prob_min"] = (
        probability_matrix.min(
            axis=1
        )
    )

    frame["prob_max"] = (
        probability_matrix.max(
            axis=1
        )
    )

    return frame


def choose_meta_model():
    return [
        (
            "Meta_LR",
            LogisticRegression(
                C=0.10,
                max_iter=4000,
                random_state=SEED,
            ),
        ),
        (
            "Meta_LR_Regularized",
            LogisticRegression(
                C=0.03,
                max_iter=4000,
                random_state=SEED,
            ),
        ),
    ]


def run_base_oof(
    X_fit,
    y_fit,
    X_eval,
    eval_name,
):
    folds = StratifiedKFold(
        n_splits=FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    model_names = list(
        build_models().keys()
    )

    oof = np.zeros(
        (
            len(X_fit),
            len(model_names),
        ),
        dtype=float,
    )

    eval_probability = np.zeros(
        (
            len(X_eval),
            len(model_names),
        ),
        dtype=float,
    )

    for fold_no, (
        train_idx,
        valid_idx,
    ) in enumerate(
        folds.split(
            X_fit,
            y_fit,
        ),
        start=1,
    ):

        print(
            f"\n  {eval_name} fold "
            f"{fold_no}/{FOLDS}"
        )

        X_fold_train = (
            X_fit.iloc[
                train_idx
            ]
        )

        X_fold_valid = (
            X_fit.iloc[
                valid_idx
            ]
        )

        y_fold_train = (
            y_fit.iloc[
                train_idx
            ]
        )

        (
            X_train_encoded,
            X_valid_encoded,
            _,
            _,
        ) = fit_transform_fold(
            X_fold_train,
            y_fold_train,
            X_fold_valid,
        )

        # Encode evaluation data using fold-train mapping
        categorical_columns = (
            get_categorical_columns(
                X_fold_train
            )
        )

        encoder = fit_target_encoder(
            X_fold_train,
            y_fold_train,
            categorical_columns,
            smoothing=20.0,
        )

        X_eval_encoded = (
            transform_target_encoded(
                X_eval,
                encoder,
            )
        )

        imputer = SimpleImputer(
            strategy="median"
        )

        X_train_encoded = pd.DataFrame(
            imputer.fit_transform(
                X_train_encoded
            ),
            columns=X_train_encoded.columns,
            index=X_train_encoded.index,
        )

        X_valid_encoded = pd.DataFrame(
            imputer.transform(
                X_valid_encoded
            ),
            columns=X_train_encoded.columns,
            index=X_valid_encoded.index,
        )

        X_eval_encoded = pd.DataFrame(
            imputer.transform(
                X_eval_encoded
            ),
            columns=X_train_encoded.columns,
            index=X_eval_encoded.index,
        )

        fitted = fit_models_for_fold(
            X_train_encoded,
            y_fold_train,
        )

        for model_position, model_name in enumerate(
            model_names
        ):

            oof[
                valid_idx,
                model_position,
            ] = (
                fitted[model_name]
                .predict_proba(
                    X_valid_encoded
                )[:, 1]
            )

            eval_probability[
                :,
                model_position,
            ] += (
                fitted[model_name]
                .predict_proba(
                    X_eval_encoded
                )[:, 1]
                / FOLDS
            )

    return (
        oof,
        eval_probability,
        model_names,
    )


def fit_final_base_models(
    X_dev,
    y_dev,
    X_test,
):
    (
        X_dev_encoded,
        X_test_encoded,
        encoder,
        imputer,
    ) = fit_transform_fold(
        X_dev,
        y_dev,
        X_test,
    )

    models = build_models()

    fitted = {}

    for name, model in models.items():

        print(
            f"  Final base model: "
            f"{name}"
        )

        model.fit(
            X_dev_encoded,
            y_dev,
        )

        fitted[name] = model

    test_probability = np.column_stack(
        [
            fitted[name]
            .predict_proba(
                X_test_encoded
            )[:, 1]
            for name in models
        ]
    )

    return (
        fitted,
        test_probability,
        encoder,
        imputer,
    )


def evaluate_meta_candidates(
    oof_base,
    validation_base,
    y_train,
    y_validation,
):
    oof_meta = make_meta_features(
        oof_base
    )

    validation_meta = make_meta_features(
        validation_base
    )

    candidates = []

    for name, model in choose_meta_model():

        print(
            f"\nTraining {name} ..."
        )

        model.fit(
            oof_meta,
            y_train,
        )

        validation_probability = (
            model.predict_proba(
                validation_meta
            )[:, 1]
        )

        # Optional calibrated version.
        from sklearn.calibration import (
            CalibratedClassifierCV
        )

        calibrated = (
            CalibratedClassifierCV(
                estimator=model,
                method="sigmoid",
                cv=4,
            )
        )

        calibrated.fit(
            oof_meta,
            y_train,
        )

        calibrated_probability = (
            calibrated.predict_proba(
                validation_meta
            )[:, 1]
        )

        for variant, probability, fitted in [
            (
                "raw",
                validation_probability,
                model,
            ),
            (
                "calibrated",
                calibrated_probability,
                calibrated,
            ),
        ]:

            best = None

            for threshold in np.arange(
                0.30,
                0.701,
                0.01,
            ):

                metrics = calculate_metrics(
                    y_validation,
                    probability,
                    float(threshold),
                )

                score = objective(
                    metrics,
                    REFERENCE,
                )

                record = {
                    "MetaModel": name,
                    "Variant": variant,
                    "Threshold": float(
                        threshold
                    ),
                    **metrics,
                    "AllAbove": score[0],
                    "MinRatio": score[1],
                    "MeanRatio": score[2],
                }

                if best is None:
                    best = record
                    best_fitted = fitted
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
                        best_fitted = fitted

            candidates.append(
                {
                    "record": best,
                    "model": best_fitted,
                    "variant": variant,
                    "meta_name": name,
                }
            )

            print(
                f"  {name} {variant}: "
                f"ValAcc="
                f"{best['Accuracy']:.4f} "
                f"ValPrec="
                f"{best['Precision']:.4f} "
                f"ValRec="
                f"{best['Recall']:.4f} "
                f"ValF1="
                f"{best['F1-score']:.4f} "
                f"ValAUC="
                f"{best['ROC-AUC']:.4f} "
                f"Threshold="
                f"{best['Threshold']:.2f}"
            )

    candidates.sort(
        key=lambda x: (
            x["record"]["AllAbove"],
            x["record"]["MinRatio"],
            x["record"]["MeanRatio"],
            x["record"]["F1-score"],
            x["record"]["ROC-AUC"],
        ),
        reverse=True,
    )

    return (
        candidates[0],
        oof_meta,
        validation_meta,
    )


def main():
    start = time.time()

    print("=" * 100)
    print(
        "SH-DAS CHAMPION REFINEMENT"
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

    X, y = prepare_target(
        df
    )

    # EXACT CANONICAL 60/20/20 SPLIT
    X_train, X_temp, y_train, y_temp = (
        train_test_split(
            X,
            y,
            test_size=0.40,
            stratify=y,
            random_state=SEED,
        )
    )

    X_validation, X_test, y_validation, y_test = (
        train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            stratify=y_temp,
            random_state=SEED,
        )
    )

    X_train = X_train.reset_index(
        drop=True
    )

    y_train = y_train.reset_index(
        drop=True
    )

    X_validation = (
        X_validation.reset_index(
            drop=True
        )
    )

    y_validation = (
        y_validation.reset_index(
            drop=True
        )
    )

    X_test = X_test.reset_index(
        drop=True
    )

    y_test = y_test.reset_index(
        drop=True
    )

    print(
        f"\nTrain:      {len(X_train):,}"
    )

    print(
        f"Validation: {len(X_validation):,}"
    )

    print(
        f"Test:       {len(X_test):,}"
    )

    print(
        f"Feature count after engineering: "
        f"{len(X.columns)}"
    )

    print(
        "\nStage 1: OOF BASE MODELS"
    )

    print(
        "-" * 100
    )

    (
        oof_base,
        validation_base,
        model_names,
    ) = run_base_oof(
        X_train,
        y_train,
        X_validation,
        "validation"
    )

    print(
        "\nBASE MODEL OOF AUC"
    )

    print(
        "-" * 100
    )

    for position, name in enumerate(
        model_names
    ):
        auc = roc_auc_score(
            y_train,
            oof_base[
                :,
                position,
            ],
        )

        print(
            f"{name:<25}"
            f"AUC={auc:.4f}"
        )

    print(
        "\nStage 2: META-MODEL "
        "VALIDATION SELECTION"
    )

    print(
        "-" * 100
    )

    (
        winner,
        train_meta,
        validation_meta,
    ) = evaluate_meta_candidates(
        oof_base,
        validation_base,
        y_train,
        y_validation,
    )

    selected_record = (
        winner["record"]
    )

    selected_threshold = float(
        selected_record[
            "Threshold"
        ]
    )

    print(
        "\nSELECTED REFINEMENT CONFIGURATION"
    )

    print(
        "-" * 100
    )

    print(
        f"Meta-model: "
        f"{winner['meta_name']}"
    )

    print(
        f"Variant: "
        f"{winner['variant']}"
    )

    print(
        f"Threshold: "
        f"{selected_threshold:.2f}"
    )

    print(
        f"Validation Accuracy: "
        f"{selected_record['Accuracy']:.4f}"
    )

    print(
        f"Validation Precision: "
        f"{selected_record['Precision']:.4f}"
    )

    print(
        f"Validation Recall: "
        f"{selected_record['Recall']:.4f}"
    )

    print(
        f"Validation F1: "
        f"{selected_record['F1-score']:.4f}"
    )

    print(
        f"Validation ROC-AUC: "
        f"{selected_record['ROC-AUC']:.4f}"
    )

    # ---------------------------------------------------------
    # FINAL DEVELOPMENT OOF
    # ---------------------------------------------------------

    print(
        "\nStage 3: CROSS-FITTED DEVELOPMENT "
        "REFIT"
    )

    print(
        "-" * 100
    )

    X_development = pd.concat(
        [
            X_train,
            X_validation,
        ],
        ignore_index=True,
    )

    y_development = pd.concat(
        [
            y_train,
            y_validation,
        ],
        ignore_index=True,
    )

    (
        development_oof,
        unused_eval,
        final_model_names,
    ) = run_base_oof(
        X_development,
        y_development,
        X_test,
        "development"
    )

    development_meta = (
        make_meta_features(
            development_oof
        )
    )

    # Train final meta architecture.
    if (
        winner["meta_name"]
        == "Meta_LR"
    ):
        base_meta = LogisticRegression(
            C=0.10,
            max_iter=4000,
            random_state=SEED,
        )
    else:
        base_meta = LogisticRegression(
            C=0.03,
            max_iter=4000,
            random_state=SEED,
        )

    if winner["variant"] == "calibrated":

        from sklearn.calibration import (
            CalibratedClassifierCV
        )

        final_meta = (
            CalibratedClassifierCV(
                estimator=base_meta,
                method="sigmoid",
                cv=4,
            )
        )

    else:
        final_meta = base_meta

    final_meta.fit(
        development_meta,
        y_development,
    )

    # Final base models on all development data.
    (
        final_base_models,
        test_base_probability,
        final_encoder,
        final_imputer,
    ) = fit_final_base_models(
        X_development,
        y_development,
        X_test,
    )

    test_meta = make_meta_features(
        test_base_probability
    )

    final_probability = (
        final_meta.predict_proba(
            test_meta
        )[:, 1]
    )

    final_metrics = calculate_metrics(
        y_test,
        final_probability,
        selected_threshold,
    )

    canonical_dominance = sum(
        final_metrics[m]
        > REFERENCE[m]
        for m in REFERENCE
    )

    super_dominance = sum(
        final_metrics[m]
        > SUPER_LEARNER_REFERENCE[m]
        for m in SUPER_LEARNER_REFERENCE
    )

    print(
        "\n" + "=" * 100
    )

    print(
        "FINAL CHAMPION REFINEMENT RESULT"
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
        f"REFINEMENT vs SUPER-LEARNER: "
        f"{super_dominance}/5"
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    comparison = pd.DataFrame(
        [
            {
                "Model": "SH-DAS Champion Refinement",
                "Type": "Refined proposed model",
                **final_metrics,
                "Threshold": selected_threshold,
                "CanonicalDominance": canonical_dominance,
            },
            {
                "Model": "SH-DAS Super-Learner",
                "Type": "Previous proposed model",
                **SUPER_LEARNER_REFERENCE,
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
        / "champion_refinement_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    summary = {
        "selected_meta_model": (
            winner["meta_name"]
        ),
        "variant": winner["variant"],
        "threshold": selected_threshold,
        "engineered_feature_count": int(
            len(X.columns)
        ),
        "final_metrics": final_metrics,
        "canonical_reference": REFERENCE,
        "super_learner_reference": (
            SUPER_LEARNER_REFERENCE
        ),
        "canonical_dominance": int(
            canonical_dominance
        ),
        "beats_all_five": bool(
            canonical_dominance == 5
        ),
        "refinement_vs_super_learner": int(
            super_dominance
        ),
        "train_size": int(
            len(X_train)
        ),
        "validation_size": int(
            len(X_validation)
        ),
        "development_size": int(
            len(X_development)
        ),
        "test_size": int(
            len(X_test)
        ),
        "folds": FOLDS,
        "runtime_seconds": float(
            time.time() - start
        ),
    }

    summary_path = (
        REPORT_DIR
        / "champion_refinement_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "SH-DAS CHAMPION REFINEMENT",
        "=" * 80,
        "",
        f"Meta-model: {winner['meta_name']}",
        f"Variant: {winner['variant']}",
        f"Threshold: {selected_threshold:.4f}",
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
            f"Refinement vs Super-Learner: "
            f"{super_dominance}/5",
            "",
            f"Runtime: "
            f"{time.time() - start:.2f} seconds",
        ]
    )

    report_path = (
        REPORT_DIR
        / "champion_refinement_report.txt"
    )

    report_path.write_text(
        "\n".join(
            report_lines
        ),
        encoding="utf-8",
    )

    bundle_path = (
        REPORT_DIR
        / "sh_das_champion_refinement_bundle.joblib"
    )

    joblib.dump(
        {
            "base_models": final_base_models,
            "meta_model": final_meta,
            "threshold": selected_threshold,
            "model_names": final_model_names,
            "seed": SEED,
        },
        bundle_path,
    )

    print(
        "\nSAVED"
    )

    print("-" * 100)
    print(comparison_path)
    print(summary_path)
    print(report_path)
    print(bundle_path)


if __name__ == "__main__":
    main()
