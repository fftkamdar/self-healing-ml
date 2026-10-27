"""
LIVE DRIFT & SELF-HEALING SIMULATOR — V2

Interactive laboratory for:
1. Creating controlled feature drift.
2. Measuring feature-level PSI drift.
3. Measuring baseline performance under the drift.
4. Running a controlled adaptive-recovery experiment.
5. Applying a conservative performance gate before accepting recovery.

This module does NOT overwrite or deploy any saved model.

Run standalone from the project root with:
    streamlit run src/live_drift_simulator.py

Or import `page_live_drift_simulator` into the main dashboard.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd
import streamlit as st

try:
    from sklearn.base import clone
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
except Exception:  # pragma: no cover
    clone = train_test_split = None
    accuracy_score = f1_score = precision_score = recall_score = roc_auc_score = None


# -----------------------------------------------------------------------------
# PROJECT PATHS
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
BASELINE_MODEL_PATH = PROJECT_ROOT / "models" / "baseline_churn_pipeline.joblib"
AUDIT_DIR = PROJECT_ROOT / "reports" / "live_drift_simulator"
LATEST_DECISION_PATH = AUDIT_DIR / "latest_recovery_decision.json"


# Project PSI heuristics used elsewhere in the dashboard.
PSI_LOW_MAX = 0.10
PSI_MODERATE_MAX = 0.25

RANDOM_STATE = 42


# -----------------------------------------------------------------------------
# DATA / MODEL HELPERS
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


@st.cache_resource(show_spinner=False)
def load_baseline_model(path: str):
    import joblib

    return joblib.load(path)


def numeric_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"Churn"}
    return [c for c in df.select_dtypes(include=np.number).columns if c not in excluded]


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Calculate PSI using quantile bins learned from the reference data."""
    ref = safe_numeric(reference)
    cur = safe_numeric(current)

    if ref.empty or cur.empty:
        return float("nan")
    if ref.nunique() <= 1:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(ref.quantile(quantiles).to_numpy())

    if len(edges) < 3:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_bins = pd.cut(ref, bins=edges, include_lowest=True)
    cur_bins = pd.cut(cur, bins=edges, include_lowest=True)

    ref_pct = ref_bins.value_counts(sort=False, normalize=True).to_numpy(dtype=float)
    cur_pct = cur_bins.value_counts(sort=False, normalize=True).to_numpy(dtype=float)

    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def classify_psi(max_psi: float) -> tuple[str, str]:
    if np.isnan(max_psi):
        return "UNKNOWN", "Not enough numeric data to assess drift."
    if max_psi < PSI_LOW_MAX:
        return "LOW", "Continue monitoring; no recovery evaluation is recommended."
    if max_psi < PSI_MODERATE_MAX:
        return "MODERATE", "Investigate affected features before triggering recovery."
    return "HIGH", "Trigger the recovery evaluation stage."


def apply_drift(
    df: pd.DataFrame,
    tenure_multiplier: float,
    monthly_charges_multiplier: float,
) -> pd.DataFrame:
    """Apply controlled drift only to the current/evaluation window."""
    drifted = df.copy()

    if "tenure" in drifted.columns:
        drifted["tenure"] = (
            drifted["tenure"] * tenure_multiplier
        ).clip(lower=0, upper=72)

    if "MonthlyCharges" in drifted.columns:
        drifted["MonthlyCharges"] = (
            drifted["MonthlyCharges"] * monthly_charges_multiplier
        ).clip(lower=0, upper=120)

    return drifted


def compute_drift_table(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    for feature in numeric_columns(reference):
        rows.append(
            {
                "Feature": feature,
                "Reference Mean": reference[feature].mean(),
                "Current Mean": current[feature].mean(),
                "Mean Shift": current[feature].mean() - reference[feature].mean(),
                "PSI": psi_numeric(reference[feature], current[feature]),
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("PSI", ascending=False).reset_index(drop=True)
    return result


def positive_label_mask(series: pd.Series) -> pd.Series:
    """Map common binary positive labels to 1 and everything else to 0."""
    normalized = series.astype(str).str.strip().str.lower()
    positive = {"yes", "1", "true", "churn", "positive"}
    return normalized.isin(positive).astype(int)


def get_model_metrics(model, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Robust binary classification metrics for Yes/No or 0/1 labels."""
    if accuracy_score is None:
        raise RuntimeError("scikit-learn is unavailable.")

    predictions = model.predict(x)
    y_true_bin = positive_label_mask(y)

    pred_series = pd.Series(predictions, index=y.index)
    pred_str = pred_series.astype(str).str.strip().str.lower()
    y_pred_bin = pred_str.isin({"yes", "1", "true", "churn", "positive"}).astype(int)

    # Explicit numeric handling for 0/1 model outputs.
    numeric_pred = pd.to_numeric(pred_series, errors="coerce")
    if numeric_pred.notna().all():
        unique_numeric = set(numeric_pred.astype(int).unique())
        if unique_numeric.issubset({0, 1}):
            y_pred_bin = numeric_pred.astype(int)

    accuracy = accuracy_score(y_true_bin, y_pred_bin)
    precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

    auc = float("nan")
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(x)
            classes = getattr(model, "classes_", None)

            if classes is not None and len(classes) == 2:
                class_strings = [str(c).strip().lower() for c in classes]
                positive_candidates = {"yes", "1", "true", "churn", "positive"}

                positive_indices = [
                    i for i, label in enumerate(class_strings)
                    if label in positive_candidates
                ]
                positive_index = positive_indices[0] if positive_indices else 1
                auc = roc_auc_score(y_true_bin, probabilities[:, positive_index])
        except Exception:
            auc = float("nan")

    return {
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1-score": float(f1),
        "ROC-AUC": float(auc),
    }


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.drop(columns=["Churn"], errors="ignore").copy()
    x = x.drop(columns=["customerID"], errors="ignore")
    return x


def save_recovery_decision(
    comparison: pd.DataFrame,
    gates: dict[str, bool],
    decision: str,
    max_psi: float,
    top_feature: str,
    tenure_multiplier: float,
    charges_multiplier: float,
) -> None:
    """Persist the latest simulator decision for dashboard audit/registry use."""
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "decision": decision,
        "max_psi": float(max_psi),
        "top_drift_feature": top_feature,
        "tenure_multiplier": float(tenure_multiplier),
        "monthly_charges_multiplier": float(charges_multiplier),
        "gates": {k: bool(v) for k, v in gates.items()},
        "comparison": comparison.to_dict(orient="records"),
        "model_modified": False,
        "model_deployed": False,
    }

    LATEST_DECISION_PATH.write_text(
        json.dumps(record, indent=2),
        encoding="utf-8",
    )


def run_recovery_experiment(
    baseline_model,
    drifted_current: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, bool], str]:
    """
    Train an adaptive candidate on half of the drifted window and evaluate both
    previous and candidate models on the untouched other half.

    Gate:
      - F1 must not decrease.
      - ROC-AUC must not decrease.
      - Recall must improve strictly.

    Returns comparison table, gate results, and final decision.
    """
    if clone is None or train_test_split is None:
        raise RuntimeError("Required scikit-learn utilities are unavailable.")
    if "Churn" not in drifted_current.columns:
        raise RuntimeError("Churn column is required for recovery evaluation.")

    adapt_df, eval_df = train_test_split(
        drifted_current,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=drifted_current["Churn"],
    )

    baseline_metrics = get_model_metrics(
        baseline_model,
        build_features(eval_df),
        eval_df["Churn"],
    )

    candidate = clone(baseline_model)
    candidate.fit(build_features(adapt_df), adapt_df["Churn"])

    candidate_metrics = get_model_metrics(
        candidate,
        build_features(eval_df),
        eval_df["Churn"],
    )

    comparison = pd.DataFrame(
        [
            {"Model": "Previous Baseline", "Stage": "Drifted Evaluation", **baseline_metrics},
            {"Model": "Adaptive Recovery Candidate", "Stage": "Drifted Evaluation", **candidate_metrics},
        ]
    )

    f1_pass = candidate_metrics["F1-score"] >= baseline_metrics["F1-score"]
    auc_pass = (
        np.isfinite(candidate_metrics["ROC-AUC"])
        and np.isfinite(baseline_metrics["ROC-AUC"])
        and candidate_metrics["ROC-AUC"] >= baseline_metrics["ROC-AUC"]
    )
    recall_pass = candidate_metrics["Recall"] > baseline_metrics["Recall"]

    gates = {
        "F1 gate": f1_pass,
        "ROC-AUC gate": auc_pass,
        "Recall improvement gate": recall_pass,
    }

    decision = "ACCEPT_RECOVERY" if all(gates.values()) else "RETAIN_PREVIOUS_MODEL"
    return comparison, gates, decision


# -----------------------------------------------------------------------------
# PAGE
# -----------------------------------------------------------------------------
def page_live_drift_simulator() -> None:
    st.markdown(
        '<div class="shml-section-title">Live Drift &amp; Self-Healing Simulator</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="shml-section-sub">Interactive laboratory for drift detection and controlled recovery. '
        'Saved models are never overwritten or deployed from this page.</div>',
        unsafe_allow_html=True,
    )

    if not DATA_PATH.exists():
        st.error(f"Dataset not found: {DATA_PATH}")
        return

    if not BASELINE_MODEL_PATH.exists():
        st.error(f"Baseline model not found: {BASELINE_MODEL_PATH}")
        return

    if accuracy_score is None:
        st.error("scikit-learn is required for this simulator.")
        return

    try:
        df = load_dataset(str(DATA_PATH))
        model = load_baseline_model(str(BASELINE_MODEL_PATH))
    except Exception as exc:
        st.error(f"Unable to load the dataset/model: {exc}")
        return

    st.caption(f"Dataset loaded: {len(df):,} rows × {len(df.columns):,} columns")

    if len(df) < 200:
        st.warning("The dataset is too small for the intended recovery simulation.")
        return

    split_idx = int(len(df) * 0.70)
    reference = df.iloc[:split_idx].copy()
    current_original = df.iloc[split_idx:].copy()

    # ------------------------------------------------------------------
    # 1. DRIFT CONTROLS
    # ------------------------------------------------------------------
    st.markdown('<div class="shml-section-title">1. Create Controlled Drift</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        tenure_multiplier = st.slider(
            "Tenure multiplier",
            1.00,
            1.50,
            1.00,
            0.05,
            help="Applied only to the current/recent window.",
        )
    with c2:
        charges_multiplier = st.slider(
            "MonthlyCharges multiplier",
            1.00,
            1.20,
            1.00,
            0.02,
            help="Applied only to the current/recent window.",
        )

    drifted_current = apply_drift(
        current_original,
        tenure_multiplier=tenure_multiplier,
        monthly_charges_multiplier=charges_multiplier,
    )

    # ------------------------------------------------------------------
    # 2. DRIFT DIAGNOSIS
    # ------------------------------------------------------------------
    st.markdown('<div class="shml-section-title">2. Drift Diagnosis</div>', unsafe_allow_html=True)
    drift = compute_drift_table(reference, drifted_current)

    if drift.empty:
        st.warning("No numeric features were available for PSI analysis.")
        return

    max_psi = float(drift["PSI"].max())
    severity, action = classify_psi(max_psi)
    top_feature = str(drift.iloc[0]["Feature"])
    top_psi = float(drift.iloc[0]["PSI"])

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Maximum PSI", f"{max_psi:.4f}")
    with m2:
        st.metric("Drift Severity", severity)
    with m3:
        st.metric("Most Affected Feature", top_feature)

    if severity == "HIGH":
        st.error(f"HIGH DRIFT — recovery evaluation is eligible. Top PSI: {top_psi:.4f}")
    elif severity == "MODERATE":
        st.warning(f"MODERATE DRIFT — investigate before recovery. Top PSI: {top_psi:.4f}")
    elif severity == "LOW":
        st.success(f"LOW DRIFT — continue monitoring. Top PSI: {top_psi:.4f}")
    else:
        st.info(action)

    shown = drift.head(10).copy()
    st.dataframe(
        shown.style.format(
            {
                "Reference Mean": "{:.3f}",
                "Current Mean": "{:.3f}",
                "Mean Shift": "{:.3f}",
                "PSI": "{:.4f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.bar_chart(shown[["Feature", "PSI"]].set_index("Feature"))

    # ------------------------------------------------------------------
    # 3. BASELINE PERFORMANCE UNDER DRIFT
    # ------------------------------------------------------------------
    st.markdown('<div class="shml-section-title">3. Baseline Performance Under the Simulated Drift</div>', unsafe_allow_html=True)

    try:
        current_original_x = build_features(current_original)
        drifted_x = build_features(drifted_current)
        y_current = current_original["Churn"]

        before = get_model_metrics(model, current_original_x, y_current)
        after = get_model_metrics(model, drifted_x, y_current)

        comparison = pd.DataFrame(
            [
                {"Stage": "Before simulated drift", **before},
                {"Stage": "After simulated drift", **after},
            ]
        )

        st.dataframe(
            comparison.style.format(
                {
                    "Accuracy": "{:.4f}",
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1-score": "{:.4f}",
                    "ROC-AUC": "{:.4f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

        metric = st.selectbox(
            "Metric to visualize",
            ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
            index=3,
            key="live_sim_metric_v2",
        )
        st.bar_chart(comparison.set_index("Stage")[[metric]])

    except Exception as exc:
        st.warning(f"Drift analysis worked, but baseline performance could not be calculated: {exc}")

    # ------------------------------------------------------------------
    # 4. SELF-HEALING DECISION + RECOVERY EVALUATION
    # ------------------------------------------------------------------
    st.markdown('<div class="shml-section-title">4. Self-Healing Recovery Evaluation</div>', unsafe_allow_html=True)

    if severity != "HIGH":
        if severity == "MODERATE":
            st.warning("Recovery evaluation is locked while drift is MODERATE. Increase drift to HIGH to demonstrate the recovery stage.")
        else:
            st.success("Recovery evaluation is not triggered because drift is below the recovery threshold.")
    else:
        st.markdown(
            """
            <div class="shml-info">
            <strong>Recovery Trigger: YES</strong><br>
            The simulated distribution shift crossed the recovery threshold. The
            candidate will be trained only on an adaptation portion of the drifted
            window and evaluated on a separate untouched evaluation portion.
            </div>
            """,
            unsafe_allow_html=True,
        )

        run_recovery = st.button(
            "Run Recovery Evaluation",
            type="primary",
            width="stretch",
            key="run_live_recovery_v2",
            help="Runs a controlled adaptive-retraining candidate and applies a conservative performance gate.",
        )

        if run_recovery:
            with st.spinner("Training and evaluating the adaptive recovery candidate..."):
                try:
                    recovery_comparison, gates, decision = run_recovery_experiment(
                        model,
                        drifted_current,
                    )
                except Exception as exc:
                    st.error(f"Recovery evaluation failed: {exc}")
                    return

            st.markdown('<div class="shml-section-title">Recovery Candidate vs Previous Model</div>', unsafe_allow_html=True)
            st.dataframe(
                recovery_comparison.style.format(
                    {
                        "Accuracy": "{:.4f}",
                        "Precision": "{:.4f}",
                        "Recall": "{:.4f}",
                        "F1-score": "{:.4f}",
                        "ROC-AUC": "{:.4f}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

            st.markdown('<div class="shml-section-title">Performance Gate</div>', unsafe_allow_html=True)
            gate_df = pd.DataFrame(
                [{"Gate Check": name, "Result": "PASS" if passed else "FAIL"} for name, passed in gates.items()]
            )
            st.dataframe(gate_df, width="stretch", hide_index=True)

            save_recovery_decision(
                recovery_comparison,
                gates,
                decision,
                max_psi,
                top_feature,
                tenure_multiplier,
                charges_multiplier,
            )

            if decision == "ACCEPT_RECOVERY":
                st.success(
                    "RECOVERY ACCEPTED — the adaptive candidate satisfied all performance gates. "
                    "The simulator does not deploy or overwrite the saved model."
                )
            else:
                st.error(
                    "RECOVERY REJECTED — the previous baseline is retained because the candidate "
                    "did not satisfy every performance gate."
                )

            st.markdown(
                f"""
                <div class="shml-info">
                <strong>Final decision:</strong> {decision}<br>
                <strong>Safety property:</strong> no saved model was modified, replaced, or deployed.<br>
                <strong>Audit record:</strong> {LATEST_DECISION_PATH.relative_to(PROJECT_ROOT)}
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("How the recovery experiment works"):
                st.markdown(
                    """
                    The drifted recent window is split into an adaptation half and a
                    separate evaluation half. The previous model is evaluated on the
                    untouched evaluation half. A cloned candidate is retrained only on
                    the adaptation half, then evaluated on the same untouched evaluation
                    half. Recovery is accepted only when F1 does not decrease, ROC-AUC
                    does not decrease, and recall improves strictly.
                    """
                )

    st.caption(
        "Academic prototype using controlled simulated drift. This page is an offline experiment and does not connect to production data."
    )


if __name__ == "__main__":
    try:
        st.set_page_config(
            page_title="Self-Healing ML | Live Drift Simulator",
            page_icon="◆",
            layout="wide",
        )
    except Exception:
        pass

    page_live_drift_simulator()
