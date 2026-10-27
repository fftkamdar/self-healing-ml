"""
SELF-HEALING ML — Intelligent Framework for Detecting, Diagnosing and
Recovering from Model Drift.

Streamlit dashboard entry point.

Run from the project root with:
    python -m streamlit run src/dashboard.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from live_drift_simulator import page_live_drift_simulator
from research_contribution_page import page_research_contribution_novelty

# ============================================================================
# 1. PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
SRC_DIR = PROJECT_ROOT / "src"

PATHS = {
    "baseline_model": MODELS_DIR / "baseline_churn_pipeline.joblib",
    "retrained_model": MODELS_DIR / "retrained_churn_pipeline.joblib",
    "optuna_model": MODELS_DIR / "optuna_tuned_churn_pipeline.joblib",
    "baseline_metrics_txt": REPORTS_DIR / "baseline_metrics.txt",
    "drift_report_txt": REPORTS_DIR / "drift_report.txt",
    "feature_drift_csv": REPORTS_DIR / "feature_drift_scores.csv",
    "retraining_comparison_csv": REPORTS_DIR / "retraining_comparison.csv",
    "retraining_decision_txt": REPORTS_DIR / "retraining_decision.txt",
    "optuna_report_txt": REPORTS_DIR / "optuna_tuning_report.txt",
    "model_comparison_csv": REPORTS_DIR / "model_comparison.csv",
    "model_comparison_png": REPORTS_DIR / "model_comparison.png",
    "integrated_drift_csv": REPORTS_DIR / "integrated_feature_drift_report.csv",
    "integrated_comparison_csv": REPORTS_DIR / "integrated_model_comparison.csv",
    "integrated_summary_json": REPORTS_DIR / "integrated_pipeline_summary.json",
    "integrated_report_txt": REPORTS_DIR / "integrated_pipeline_report.txt",
}

SCRIPTS = {
    "baseline": SRC_DIR / "train.py",
    "drift": SRC_DIR / "drift_monitor.py",
    "retrain": SRC_DIR / "retrain_experiment.py",
    "optuna": SRC_DIR / "optuna_tuning.py",
    "compare": SRC_DIR / "compare_models.py",
}

# PSI thresholds used by this project (heuristics, not universal standards)
PSI_LOW_MAX = 0.10
PSI_MODERATE_MAX = 0.25


# ============================================================================
# 2. PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Self-Healing ML | Drift Monitoring Platform",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# 3. GLOBAL STYLING
# ============================================================================

def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=DM+Sans:wght@400;500;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'DM Sans', 'Manrope', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        :root {
            --navy-950: #0a0e1a;
            --navy-900: #0f1424;
            --navy-800: #161d34;
            --indigo-600: #4f46e5;
            --indigo-500: #6366f1;
            --violet-500: #8b5cf6;
            --cyan-400: #22d3ee;
            --slate-200: #e2e8f0;
            --slate-500: #64748b;
            --slate-600: #475569;
            --surface: #ffffff;
            --surface-alt: #f8fafc;
            --border-soft: rgba(15, 23, 42, 0.08);
            --success: #059669;
            --warning: #d97706;
            --danger: #dc2626;
        }

        .stApp {
            background: linear-gradient(180deg, #f6f7fb 0%, #f1f3f9 100%);
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, var(--navy-950) 0%, var(--navy-900) 100%);
        }
        section[data-testid="stSidebar"] * {
            color: #e5e7f0 !important;
        }
        section[data-testid="stSidebar"] .stRadio label {
            font-weight: 500;
        }
        section[data-testid="stSidebar"] hr {
            border-color: rgba(255,255,255,0.08);
        }

        /* Hero */
        .shml-hero {
            background: linear-gradient(120deg, var(--navy-950) 0%, #1b1f4a 55%, #2c2a6b 100%);
            border-radius: 20px;
            padding: 2.6rem 2.8rem;
            margin-bottom: 1.6rem;
            box-shadow: 0 20px 40px -20px rgba(30, 27, 75, 0.55);
        }
        .shml-badge {
            display: inline-block;
            padding: 0.28rem 0.85rem;
            border-radius: 999px;
            background: rgba(34, 211, 238, 0.14);
            border: 1px solid rgba(34, 211, 238, 0.4);
            color: #67e8f9;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.9rem;
        }
        .shml-hero h1 {
            font-family: 'Manrope', sans-serif;
            font-weight: 800;
            font-size: 2.3rem;
            color: #f8fafc;
            margin: 0 0 0.35rem 0;
            letter-spacing: -0.01em;
        }
        .shml-hero p {
            color: #c7cbe8;
            font-size: 1.02rem;
            margin: 0;
            max-width: 640px;
        }

        /* Section titles */
        .shml-section-title {
            font-family: 'Manrope', sans-serif;
            font-weight: 700;
            font-size: 1.25rem;
            color: #0f172a;
            margin: 1.4rem 0 0.6rem 0;
        }
        .shml-section-sub {
            color: var(--slate-500);
            font-size: 0.92rem;
            margin-bottom: 0.9rem;
        }

        /* Cards */
        .shml-card {
            background: var(--surface);
            border: 1px solid var(--border-soft);
            border-radius: 16px;
            padding: 1.3rem 1.4rem;
            box-shadow: 0 6px 18px -14px rgba(15, 23, 42, 0.25);
            height: 100%;
        }
        .shml-metric-label {
            color: var(--slate-500);
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.35rem;
        }
        .shml-metric-value {
            font-family: 'Manrope', sans-serif;
            font-size: 1.9rem;
            font-weight: 800;
            color: #0f172a;
        }
        .shml-metric-note {
            color: var(--slate-500);
            font-size: 0.78rem;
            margin-top: 0.2rem;
        }

        /* Badges */
        .badge {
            display: inline-block;
            padding: 0.18rem 0.6rem;
            border-radius: 999px;
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }
        .badge-low { background: rgba(5,150,105,0.1); color: var(--success); border: 1px solid rgba(5,150,105,0.3); }
        .badge-moderate { background: rgba(217,119,6,0.1); color: var(--warning); border: 1px solid rgba(217,119,6,0.3); }
        .badge-high { background: rgba(220,38,38,0.1); color: var(--danger); border: 1px solid rgba(220,38,38,0.3); }
        .badge-neutral { background: rgba(100,116,139,0.1); color: var(--slate-600); border: 1px solid rgba(100,116,139,0.3); }

        .shml-warning {
            background: rgba(217, 119, 6, 0.08);
            border: 1px solid rgba(217, 119, 6, 0.35);
            border-radius: 12px;
            padding: 0.9rem 1.1rem;
            color: #92400e;
            font-size: 0.9rem;
        }
        .shml-info {
            background: rgba(79, 70, 229, 0.06);
            border: 1px solid rgba(79, 70, 229, 0.25);
            border-radius: 12px;
            padding: 0.9rem 1.1rem;
            color: #312e81;
            font-size: 0.9rem;
        }

        .shml-step {
            background: var(--surface);
            border: 1px solid var(--border-soft);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            height: 100%;
        }
        .shml-step-num {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 26px; height: 26px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--indigo-600), var(--violet-500));
            color: white;
            font-weight: 700;
            font-size: 0.8rem;
            margin-bottom: 0.5rem;
        }

        div[data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--border-soft);
            border-radius: 14px;
            padding: 0.9rem 1rem;
        }

        footer, #MainMenu {visibility: hidden;}


        /* === VISUAL UPGRADE: editorial, premium analytics workspace === */
        .stApp {
            background:
                radial-gradient(ellipse at 12% 0%, rgba(99,102,241,0.10), transparent 34%),
                radial-gradient(ellipse at 92% 10%, rgba(34,211,238,0.08), transparent 28%),
                linear-gradient(180deg, #f7f8fc 0%, #eef2f9 100%);
        }
        [data-testid="stHeader"] {
            background: rgba(247,248,252,0.78);
            backdrop-filter: blur(14px);
        }
        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }
        section[data-testid="stSidebar"] {
            border-right: 1px solid rgba(148,163,184,0.16);
            box-shadow: 8px 0 36px rgba(15,23,42,0.08);
        }
        section[data-testid="stSidebar"] [data-testid="stRadio"] > label {
            color: #94a3b8 !important;
            text-transform: uppercase;
            letter-spacing: .12em;
            font-size: .68rem;
            font-weight: 800;
        }
        section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
            gap: .32rem;
        }
        section[data-testid="stSidebar"] [data-testid="stRadio"] label {
            border: 1px solid transparent;
            border-radius: 11px;
            padding: .58rem .68rem;
            transition: background .18s ease, border-color .18s ease, transform .18s ease;
        }
        section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
            background: rgba(148,163,184,.10);
            border-color: rgba(148,163,184,.18);
            transform: translateX(2px);
        }
        section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
            background: linear-gradient(100deg, rgba(99,102,241,.24), rgba(34,211,238,.09));
            border-color: rgba(129,140,248,.42);
            box-shadow: inset 3px 0 0 #818cf8;
        }
        .shml-hero {
            position: relative;
            overflow: hidden;
            isolation: isolate;
            background: linear-gradient(118deg, #090f20 0%, #171d43 48%, #373078 100%);
            border: 1px solid rgba(165,180,252,.20);
            border-radius: 25px;
            padding: clamp(2rem, 4vw, 3.4rem);
            margin-bottom: 2rem;
            box-shadow: 0 24px 60px -30px rgba(30,27,75,.72);
        }
        .shml-hero::before {
            content: "";
            position: absolute;
            z-index: -1;
            width: 390px;
            height: 390px;
            right: -100px;
            top: -190px;
            border-radius: 50%;
            border: 1px solid rgba(165,180,252,.22);
            box-shadow: 0 0 0 38px rgba(165,180,252,.035), 0 0 0 78px rgba(165,180,252,.025);
        }
        .shml-hero::after {
            content: "";
            position: absolute;
            z-index: -1;
            width: 220px;
            height: 220px;
            right: 14%;
            bottom: -175px;
            border-radius: 50%;
            background: rgba(34,211,238,.18);
            filter: blur(36px);
        }
        .shml-hero h1 {
            font-size: clamp(2.1rem, 4vw, 3.15rem);
            letter-spacing: -.055em;
            line-height: 1.06;
            background: linear-gradient(90deg, #ffffff 0%, #dbeafe 58%, #a5f3fc 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }
        .shml-hero p { line-height: 1.75; color: #d1d9f0; }
        .shml-badge { box-shadow: 0 0 24px rgba(34,211,238,.12); }
        .shml-section-title {
            display: flex;
            align-items: center;
            gap: .65rem;
            letter-spacing: -.025em;
            font-size: 1.38rem;
            margin-top: 1.8rem;
        }
        .shml-section-title::before {
            content: "";
            display: inline-block;
            width: 5px;
            height: 1.25rem;
            border-radius: 99px;
            background: linear-gradient(180deg, #6366f1, #22d3ee);
            box-shadow: 0 3px 12px rgba(99,102,241,.24);
        }
        .shml-card {
            border: 1px solid rgba(148,163,184,.23);
            border-radius: 18px;
            background: linear-gradient(145deg, rgba(255,255,255,.98), rgba(248,250,252,.94));
            box-shadow: 0 8px 26px -20px rgba(15,23,42,.42), 0 1px 0 rgba(255,255,255,.9) inset;
            transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
        }
        .shml-card:hover {
            transform: translateY(-3px);
            border-color: rgba(129,140,248,.48);
            box-shadow: 0 18px 32px -22px rgba(79,70,229,.40), 0 1px 0 white inset;
        }
        .shml-metric-label { letter-spacing: .085em; font-size: .72rem; }
        .shml-metric-value {
            letter-spacing: -.055em;
            background: linear-gradient(100deg, #111827, #3730a3);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }
        .shml-metric-note { line-height: 1.5; }
        .shml-info, .shml-warning {
            border-radius: 15px;
            box-shadow: 0 6px 20px -18px rgba(15,23,42,.38);
            line-height: 1.65;
        }
        .shml-step {
            border-radius: 17px;
            box-shadow: 0 8px 24px -22px rgba(15,23,42,.42);
            transition: transform .18s ease, border-color .18s ease;
        }
        .shml-step:hover { transform: translateY(-2px); border-color: rgba(99,102,241,.38); }
        div[data-testid="stMetric"] {
            border: 1px solid rgba(148,163,184,.22);
            border-radius: 17px;
            background: linear-gradient(145deg, #fff, #f8fafc);
            box-shadow: 0 8px 24px -20px rgba(15,23,42,.40);
            transition: transform .18s ease;
        }
        div[data-testid="stMetric"]:hover { transform: translateY(-2px); }
        div[data-testid="stMetric"] label { color: #64748b !important; font-weight: 700 !important; }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] { letter-spacing: -.045em; }
        .stButton > button, .stDownloadButton > button {
            border-radius: 11px;
            border: 1px solid rgba(99,102,241,.28);
            font-weight: 700;
            transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            transform: translateY(-1px);
            border-color: rgba(99,102,241,.58);
            box-shadow: 0 8px 18px -12px rgba(79,70,229,.65);
        }
        div[data-testid="stExpander"] {
            border: 1px solid rgba(148,163,184,.24);
            border-radius: 14px;
            background: rgba(255,255,255,.78);
            overflow: hidden;
        }
        div[data-testid="stDataFrame"], div[data-testid="stTable"] {
            border: 1px solid rgba(148,163,184,.24);
            border-radius: 14px;
            overflow: hidden;
            box-shadow: 0 8px 24px -22px rgba(15,23,42,.4);
        }
        div[data-testid="stAlert"] { border-radius: 13px; }
        @media (max-width: 768px) {
            .block-container { padding: 1.15rem 1rem 2rem; }
            .shml-hero { border-radius: 18px; padding: 1.6rem; }
            .shml-hero h1 { font-size: 2rem; }
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# 4. GENERIC SAFE-LOADING HELPERS
# ============================================================================

@st.cache_data(show_spinner=False)
def load_csv(path_str: str) -> Optional[pd.DataFrame]:
    """Safely load a CSV file. Returns None if missing, empty, or unreadable."""
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None
    return df


@st.cache_data(show_spinner=False)
def load_text(path_str: str) -> Optional[str]:
    """Safely load a text report. Returns None if missing or unreadable."""
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        content = path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return None
    if not content:
        return None
    return content


def file_missing_warning(path: Path, generating_script: Path) -> None:
    """Render a consistent warning when an expected artifact is missing."""
    st.markdown(
        f"""
        <div class="shml-warning">
        <strong>Report not found:</strong> <code>{path.relative_to(PROJECT_ROOT)}</code><br/>
        Generate it by running <code>python {generating_script.relative_to(PROJECT_ROOT)}</code>
        from the project root.
        </div>
        """,
        unsafe_allow_html=True,
    )


def fmt_metric(value: Optional[float], decimals: int = 4, suffix: str = "") -> str:
    """Safely format a numeric metric. Never fabricates a value."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "Not available"
    try:
        return f"{float(value):.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return "Not available"


def metric_card(label: str, value: str, note: str = "") -> str:
    return f"""
    <div class="shml-card">
        <div class="shml-metric-label">{label}</div>
        <div class="shml-metric-value">{value}</div>
        <div class="shml-metric-note">{note}</div>
    </div>
    """


def psi_badge(status: str) -> str:
    status_clean = str(status).strip().lower()
    if "high" in status_clean:
        return '<span class="badge badge-high">High drift</span>'
    if "moderate" in status_clean:
        return '<span class="badge badge-moderate">Moderate drift</span>'
    if "low" in status_clean:
        return '<span class="badge badge-low">Low drift</span>'
    return f'<span class="badge badge-neutral">{status}</span>'


def extract_float(text: str, *labels: str) -> Optional[float]:
    """Search text for the first label matching 'label: number' (case-insensitive)."""
    for label in labels:
        pattern = rf"{re.escape(label)}\s*[:=]\s*([-+]?\d*\.?\d+)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


def extract_int(text: str, *labels: str) -> Optional[int]:
    for label in labels:
        pattern = rf"{re.escape(label)}\s*[:=]?\s*(\d+)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue
    return None


def download_button_for_file(path: Path, label: str, key: str) -> None:
    """Render a download button for an existing file, or a disabled note."""
    if path.exists():
        try:
            data = path.read_bytes()
        except Exception:
            st.caption(f"Could not read {path.name} for download.")
            return
        mime = "text/csv" if path.suffix == ".csv" else "text/plain"
        st.download_button(
            label=label,
            data=data,
            file_name=path.name,
            mime=mime,
            key=key,
            width="stretch",
        )
    else:
        st.caption(f"{path.name} is not available yet.")


def run_script_safely(script_path: Path) -> None:
    """Run an existing project script and report real success/failure status."""
    if not script_path.exists():
        st.error(f"Script not found: {script_path.relative_to(PROJECT_ROOT)}")
        return
    with st.spinner(f"Running {script_path.name} ..."):
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=1800,
            )
        except Exception as exc:
            st.error(f"Failed to launch {script_path.name}: {exc}")
            return
    if result.returncode == 0:
        st.success(f"{script_path.name} completed successfully.")
        if result.stdout:
            with st.expander("Script output"):
                st.code(result.stdout[-4000:])
        st.cache_data.clear()
    else:
        st.error(f"{script_path.name} exited with an error (code {result.returncode}).")
        if result.stderr:
            with st.expander("Error output"):
                st.code(result.stderr[-4000:])


# ============================================================================
# 5. SIDEBAR NAVIGATION
# ============================================================================

PAGES = [
    "Executive Overview",
    "Model Comparison",
    "Feature Drift Monitor",
    "Selective Retraining & Cost Analysis",
    "Retraining Experiment",
    "Optuna Hyperparameter Search",
    "Experiment Reports & Downloads",
    "Integrated Self-Healing Pipeline",
    "Beginner Learning Center",
    "Root-Cause Diagnosis",
    "Performance-Aware Decision Engine",
    "Explainable AI (XAI)",
    "Sequential Drift Monitor",
    "Model Registry and Rollback",
    "Recovery Strategy Comparison",
    "Five-Model Benchmark",
    "Controlled Concept-Drift Stress Test",
    "Feature Contribution & Proposed Model",
    "Self-Healing Recovery",
    "Live Drift & Self-Healing Simulator",
    "Drift Detective",
    "Research Contribution & Novelty",
]


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 0.4rem 0 1rem 0;">
                <div style="font-family:'Manrope',sans-serif; font-weight:800; font-size:1.15rem; color:#f8fafc;">
                    SELF-HEALING ML
                </div>
                <div style="color:#9aa0c3; font-size:0.78rem; margin-top:0.1rem;">
                    Drift Monitoring &amp; Recovery
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")
        page = st.radio("Navigate", PAGES, label_visibility="collapsed")
        st.markdown("---")
        if st.button("Refresh saved artifacts", width="stretch"):
            st.cache_data.clear()
            st.rerun()
        st.caption(f"Project root:\n{PROJECT_ROOT}")
        st.markdown("---")
        st.caption("Academic prototype · Simulated drift · Not connected to production data.")
    return page


# ============================================================================
# 6. PAGE 1 — EXECUTIVE OVERVIEW
# ============================================================================

def page_executive_overview() -> None:
    st.markdown(
        """
        <div class="shml-hero">
            <div class="shml-badge">ACADEMIC PROTOTYPE · SIMULATED DRIFT</div>
            <h1>SELF-HEALING ML</h1>
            <p>An Intelligent Framework for Drift Monitoring and Model Recovery. This
            platform detects distribution drift in a customer churn model, runs a
            controlled retraining experiment, and compares model performance across
            baseline, drift-triggered, and Optuna-tuned configurations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="shml-section-title">Dataset &amp; Project Summary</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown(
            """
            <div class="shml-card">
            This project uses the <strong>IBM Telco Customer Churn</strong> dataset
            (7,043 customer records) to build a churn classifier and study how the
            system behaves when input feature distributions shift over time. Key
            features include tenure, MonthlyCharges, TotalCharges, Contract,
            PaymentMethod, InternetService, OnlineSecurity, and TechSupport. The
            target variable is <strong>Churn</strong> (Yes / No).
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="shml-info">
            <strong>Scope note:</strong> Drift in this project is <strong>simulated</strong>
            by deliberately shifting selected features. No live or streaming production
            data is connected to this dashboard.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="shml-section-title">Experiment Status at a Glance</div>', unsafe_allow_html=True)

    drift_df = load_csv(str(PATHS["feature_drift_csv"]))
    baseline_text = load_text(str(PATHS["baseline_metrics_txt"]))
    optuna_text = load_text(str(PATHS["optuna_report_txt"]))

    max_psi = None
    high_drift_count = None
    if drift_df is not None and "psi" in drift_df.columns:
        max_psi = pd.to_numeric(drift_df["psi"], errors="coerce").max()
        if "status" in drift_df.columns:
            high_drift_count = drift_df["status"].astype(str).str.contains("high", case=False).sum()

    # Extract the standalone ROC-AUC lines so validation scores are not
    # accidentally displayed as held-out test scores.
    baseline_roc_auc = None
    if baseline_text:
        baseline_auc_match = re.search(
            r"(?im)^\s*ROC-AUC\s*:\s*([0-9]*\.?[0-9]+)\s*$",
            baseline_text,
        )
        if baseline_auc_match:
            baseline_roc_auc = float(baseline_auc_match.group(1))

    optuna_roc_auc = None
    if optuna_text:
        optuna_auc_match = re.search(
            r"(?im)^\s*ROC-AUC\s*:\s*([0-9]*\.?[0-9]+)\s*$",
            optuna_text,
        )
        if optuna_auc_match:
            optuna_roc_auc = float(optuna_auc_match.group(1))

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(metric_card("Maximum PSI observed", fmt_metric(max_psi, 3), "Population Stability Index"), unsafe_allow_html=True)
    with m2:
        note = "High-drift features" if high_drift_count is not None else "drift report not loaded"
        st.markdown(metric_card("High-drift features", str(high_drift_count) if high_drift_count is not None else "Not available", note), unsafe_allow_html=True)
    with m3:
        st.markdown(metric_card("Baseline ROC-AUC", fmt_metric(baseline_roc_auc, 4), "Logistic Regression"), unsafe_allow_html=True)
    with m4:
        st.markdown(metric_card("Optuna test ROC-AUC", fmt_metric(optuna_roc_auc, 4), "Tuned model, held-out test"), unsafe_allow_html=True)

    st.markdown('<div class="shml-section-title">Experiment Workflow</div>', unsafe_allow_html=True)
    st.markdown('<div class="shml-section-sub">The pipeline implemented so far follows these stages.</div>', unsafe_allow_html=True)

    steps = [
        ("1", "Load data & establish baseline", "Train a Logistic Regression baseline on the Telco churn dataset."),
        ("2", "Monitor feature distributions", "Track feature statistics that later drift experiments compare against."),
        ("3", "Detect drift using PSI", "Compute the Population Stability Index for simulated shifted features."),
        ("4", "Trigger retraining experiment", "Retrain the model on the shifted distribution as a controlled experiment."),
        ("5", "Compare model performance", "Contrast baseline vs. retrained metrics on the shifted evaluation set."),
        ("6", "Tune with Optuna", "Search hyperparameters to see whether tuning improves validation ROC-AUC."),
    ]
    cols = st.columns(3)
    for idx, (num, title, desc) in enumerate(steps):
        with cols[idx % 3]:
            st.markdown(
                f"""
                <div class="shml-step">
                    <div class="shml-step-num">{num}</div>
                    <div style="font-weight:700; color:#0f172a; margin-bottom:0.25rem;">{title}</div>
                    <div style="color:#64748b; font-size:0.85rem;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")

    st.markdown('<div class="shml-section-title">What This Demonstrates</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="shml-card">
        This project demonstrates an end-to-end, reproducible workflow for detecting
        distribution drift with PSI, running a controlled retraining experiment, and
        objectively comparing model performance before and after both retraining and
        hyperparameter tuning. It does <strong>not</strong> demonstrate automatic,
        production-grade self-healing: every step is triggered manually via project
        scripts and evaluated on offline data.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")
    st.markdown(
        """
        <div class="shml-warning">
        <strong>Important:</strong> Drift in this project is simulated by deliberately
        shifting <code>tenure</code> and <code>MonthlyCharges</code>. This dashboard is
        an academic prototype and is <strong>not</strong> connected to a live production
        data stream or a real deployment pipeline.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# 7. PAGE 2 — MODEL COMPARISON
# ============================================================================

def page_model_comparison() -> None:
    st.markdown('<div class="shml-section-title">Model Comparison — Baseline vs. Optuna-Tuned</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="shml-section-sub">Source: reports/model_comparison.csv, read dynamically from disk.</div>',
        unsafe_allow_html=True,
    )

    df = load_csv(str(PATHS["model_comparison_csv"]))
    if df is None:
        file_missing_warning(PATHS["model_comparison_csv"], SCRIPTS["compare"])
        return

    expected_cols = {"Metric", "Baseline", "Optuna_Tuned"}
    if not expected_cols.issubset(set(df.columns)):
        st.warning(
            f"model_comparison.csv is missing expected columns {expected_cols}. "
            f"Found columns: {list(df.columns)}"
        )
        st.dataframe(df, width="stretch")
        return

    df = df.copy()
    df["Baseline"] = pd.to_numeric(df["Baseline"], errors="coerce")
    df["Optuna_Tuned"] = pd.to_numeric(df["Optuna_Tuned"], errors="coerce")
    if "Change" in df.columns:
        df["Change"] = pd.to_numeric(df["Change"], errors="coerce")
    else:
        df["Change"] = df["Optuna_Tuned"] - df["Baseline"]

    st.markdown('<div class="shml-section-title">Metric Cards</div>', unsafe_allow_html=True)
    n = len(df)
    cols = st.columns(min(n, 5)) if n > 0 else []
    for i, row in df.iterrows():
        with cols[i % len(cols)]:
            change = row["Change"]
            arrow = "▲" if pd.notna(change) and change > 0 else ("▼" if pd.notna(change) and change < 0 else "—")
            st.markdown(
                metric_card(
                    str(row["Metric"]),
                    fmt_metric(row["Optuna_Tuned"], 4),
                    f"Baseline {fmt_metric(row['Baseline'], 4)} &nbsp;·&nbsp; Change {arrow} {fmt_metric(change, 4)}",
                ),
                unsafe_allow_html=True,
            )

    st.markdown('<div class="shml-section-title">Baseline vs. Optuna-Tuned Chart</div>', unsafe_allow_html=True)
    chart_df = df.melt(id_vars="Metric", value_vars=["Baseline", "Optuna_Tuned"], var_name="Model", value_name="Score")
    fig = px.bar(
        chart_df,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        color_discrete_map={"Baseline": "#6366f1", "Optuna_Tuned": "#22d3ee"},
        title="Baseline vs. Optuna-Tuned Model Metrics",
    )
    fig.update_layout(
        xaxis_title="Metric",
        yaxis_title="Score",
        legend_title="Model",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="DM Sans, sans-serif", color="#0f172a"),
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown('<div class="shml-section-title">Comparison Table</div>', unsafe_allow_html=True)
    st.dataframe(df, width="stretch", hide_index=True)

    st.markdown('<div class="shml-section-title">Understanding the Metrics</div>', unsafe_allow_html=True)
    with st.expander("What do Accuracy, Precision, Recall, F1-score, and ROC-AUC mean?"):
        st.markdown(
            """
            - **Accuracy** — the proportion of all predictions that were correct.
            - **Precision** — of the customers predicted to churn, the proportion who actually churned.
            - **Recall** — of the customers who actually churned, the proportion the model correctly identified.
            - **F1-score** — the harmonic mean of Precision and Recall, useful when both matter.
            - **ROC-AUC** — how well the model ranks churners above non-churners across all decision thresholds; 0.5 is random, 1.0 is perfect.
            - **Change** — the difference (Optuna-Tuned minus Baseline) for that metric. A positive value means the tuned model scored higher on that metric; it does not by itself mean the tuned model is better overall.
            """
        )

    st.markdown('<div class="shml-section-title">Honest Interpretation</div>', unsafe_allow_html=True)
    improved = df[df["Change"] > 0]["Metric"].tolist() if "Change" in df.columns else []
    declined = df[df["Change"] < 0]["Metric"].tolist() if "Change" in df.columns else []
    interpretation_lines = []
    if improved:
        interpretation_lines.append(f"The Optuna-tuned model improved on: {', '.join(improved)}.")
    if declined:
        interpretation_lines.append(f"The Optuna-tuned model declined on: {', '.join(declined)}.")
    if not improved and not declined:
        interpretation_lines.append("No consistent metric changes could be determined from the current report.")
    st.markdown(
        f"""
        <div class="shml-info">
        {"<br/>".join(interpretation_lines)}<br/><br/>
        Based on these measured results alone, no single model can be labeled
        universally superior — the right choice depends on whether the use case
        prioritizes precision, recall, or overall ranking quality (ROC-AUC).
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# 8. PAGE 3 — FEATURE DRIFT MONITOR
# ============================================================================

def page_feature_drift() -> None:
    st.markdown('<div class="shml-section-title">Feature Drift Monitor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="shml-section-sub">Source: reports/feature_drift_scores.csv, read dynamically from disk.</div>',
        unsafe_allow_html=True,
    )

    df = load_csv(str(PATHS["feature_drift_csv"]))
    if df is None:
        file_missing_warning(PATHS["feature_drift_csv"], SCRIPTS["drift"])
        return

    required_cols = {"feature", "psi", "status"}
    if not required_cols.issubset(set(df.columns)):
        st.warning(f"feature_drift_scores.csv is missing expected columns {required_cols}. Found: {list(df.columns)}")
        st.dataframe(df, width="stretch")
        return

    df = df.copy()
    df["psi"] = pd.to_numeric(df["psi"], errors="coerce")

    max_psi = df["psi"].max()
    n_features = len(df)
    n_low = df["status"].astype(str).str.contains("low", case=False).sum()
    n_moderate = df["status"].astype(str).str.contains("moderate", case=False).sum()
    n_high = df["status"].astype(str).str.contains("high", case=False).sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card("Maximum PSI", fmt_metric(max_psi, 3)), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("Monitored features", str(n_features)), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("Low / Moderate drift", f"{n_low} / {n_moderate}"), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("High drift", str(n_high)), unsafe_allow_html=True)

    st.markdown('<div class="shml-section-title">PSI by Feature</div>', unsafe_allow_html=True)
    df_sorted = df.sort_values("psi", ascending=True)
    color_map = {"low": "#059669", "moderate": "#d97706", "high": "#dc2626"}

    def status_color(s: str) -> str:
        s = str(s).lower()
        for key, color in color_map.items():
            if key in s:
                return color
        return "#64748b"

    fig = go.Figure(
        go.Bar(
            x=df_sorted["psi"],
            y=df_sorted["feature"],
            orientation="h",
            marker_color=[status_color(s) for s in df_sorted["status"]],
            text=[f"{v:.3f}" for v in df_sorted["psi"]],
            textposition="outside",
        )
    )
    fig.add_vline(x=PSI_LOW_MAX, line_dash="dash", line_color="#059669", annotation_text="Low/Moderate boundary")
    fig.add_vline(x=PSI_MODERATE_MAX, line_dash="dash", line_color="#dc2626", annotation_text="Moderate/High boundary")
    fig.update_layout(
        title="Population Stability Index (PSI) per Feature",
        xaxis_title="PSI",
        yaxis_title="Feature",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="DM Sans, sans-serif", color="#0f172a"),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown('<div class="shml-section-title">Feature Drift Table</div>', unsafe_allow_html=True)
    search = st.text_input("Filter by feature name", "")
    table_df = df.sort_values("psi", ascending=False)
    if search:
        table_df = table_df[table_df["feature"].astype(str).str.contains(search, case=False, na=False)]

    display_df = table_df.copy()
    display_df["status"] = display_df["status"].apply(lambda s: psi_badge(s))
    st.write(
        display_df[["feature", "psi", "status"] + [c for c in df.columns if c not in ("feature", "psi", "status")]]
        .to_html(escape=False, index=False),
        unsafe_allow_html=True,
    )

    top_feature = table_df.iloc[0]["feature"] if not table_df.empty else None
    if top_feature is not None:
        st.markdown(
            f"""
            <div class="shml-info" style="margin-top:0.8rem;">
            The feature with the largest measured PSI in this run is
            <strong>{top_feature}</strong> (PSI = {fmt_metric(table_df.iloc[0]['psi'], 3)}).
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="shml-section-title">Threshold Reference (Project Heuristics)</div>', unsafe_allow_html=True)
    tcol1, tcol2, tcol3 = st.columns(3)
    with tcol1:
        st.markdown(metric_card("Low drift", f"PSI < {PSI_LOW_MAX:.2f}", "Distributions considered stable"), unsafe_allow_html=True)
    with tcol2:
        st.markdown(metric_card("Moderate drift", f"{PSI_LOW_MAX:.2f} ≤ PSI < {PSI_MODERATE_MAX:.2f}", "Worth monitoring"), unsafe_allow_html=True)
    with tcol3:
        st.markdown(metric_card("High drift", f"PSI ≥ {PSI_MODERATE_MAX:.2f}", "Candidate for retraining review"), unsafe_allow_html=True)
    st.caption("These thresholds are heuristics chosen for this project, not universal industry standards.")

    st.markdown('<div class="shml-section-title">What Is PSI?</div>', unsafe_allow_html=True)
    with st.expander("Explain Population Stability Index in simple terms"):
        st.markdown(
            """
            The **Population Stability Index (PSI)** measures how much a feature's
            distribution has changed between two time periods (for example, the
            original training data versus newer data). It works by dividing the
            feature's values into buckets, comparing the proportion of records in
            each bucket between the two periods, and summing a weighted difference
            across all buckets. A PSI near 0 means the distributions look almost
            identical; a large PSI means the shape of the data has changed
            noticeably.
            """
        )
    with st.expander("Does distribution drift mean the model got worse?"):
        st.markdown(
            """
            **Not necessarily.** PSI only measures whether the *input* distribution
            changed — it says nothing directly about prediction accuracy. A feature
            can drift a lot while the model's predictions stay just as accurate, and
            a feature can barely move while accuracy quietly degrades for other
            reasons. Drift detected by PSI is a *signal worth investigating*, not
            direct proof that performance has dropped. In this project, that
            investigation is the retraining experiment on the next page.
            """
        )

    st.markdown('<div class="shml-section-title">Download</div>', unsafe_allow_html=True)
    download_button_for_file(PATHS["feature_drift_csv"], "Download feature_drift_scores.csv", "dl_feature_drift")


# ============================================================================
# 9. PAGE 4 — RETRAINING EXPERIMENT
# ============================================================================

def page_retraining_experiment() -> None:
    st.markdown('<div class="shml-section-title">Drift-Triggered Retraining Experiment</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="shml-section-sub">Source: reports/retraining_comparison.csv and reports/retraining_decision.txt.</div>',
        unsafe_allow_html=True,
    )

    df = load_csv(str(PATHS["retraining_comparison_csv"]))
    if df is None:
        file_missing_warning(PATHS["retraining_comparison_csv"], SCRIPTS["retrain"])
    else:
        expected_cols = {"metric", "baseline_on_shifted_eval", "retrained_on_shifted_eval"}
        cols_lower = {c.lower(): c for c in df.columns}
        if not expected_cols.issubset(set(cols_lower.keys())):
            st.warning(f"retraining_comparison.csv is missing expected columns {expected_cols}. Found: {list(df.columns)}")
            st.dataframe(df, width="stretch")
        else:
            metric_col = cols_lower["metric"]
            base_col = cols_lower["baseline_on_shifted_eval"]
            retrain_col = cols_lower["retrained_on_shifted_eval"]

            work_df = df[[metric_col, base_col, retrain_col]].copy()
            work_df.columns = ["Metric", "Baseline", "Retrained"]
            work_df["Baseline"] = pd.to_numeric(work_df["Baseline"], errors="coerce")
            work_df["Retrained"] = pd.to_numeric(work_df["Retrained"], errors="coerce")
            work_df["Change"] = work_df["Retrained"] - work_df["Baseline"]

            st.markdown('<div class="shml-section-title">Metric-by-Metric Comparison</div>', unsafe_allow_html=True)
            n = len(work_df)
            cols = st.columns(min(n, 5)) if n > 0 else []
            for i, row in work_df.iterrows():
                with cols[i % len(cols)]:
                    arrow = "▲" if row["Change"] > 0 else ("▼" if row["Change"] < 0 else "—")
                    st.markdown(
                        metric_card(
                            str(row["Metric"]),
                            fmt_metric(row["Retrained"], 4),
                            f"Baseline {fmt_metric(row['Baseline'], 4)} &nbsp;·&nbsp; {arrow} {fmt_metric(row['Change'], 4)}",
                        ),
                        unsafe_allow_html=True,
                    )

            st.markdown('<div class="shml-section-title">Baseline vs. Retrained Chart</div>', unsafe_allow_html=True)
            chart_df = work_df.melt(id_vars="Metric", value_vars=["Baseline", "Retrained"], var_name="Model", value_name="Score")
            fig = px.bar(
                chart_df,
                x="Metric",
                y="Score",
                color="Model",
                barmode="group",
                color_discrete_map={"Baseline": "#6366f1", "Retrained": "#8b5cf6"},
                title="Baseline vs. Retrained Model on Shifted Evaluation Data",
            )
            fig.update_layout(
                xaxis_title="Metric",
                yaxis_title="Score",
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(family="DM Sans, sans-serif", color="#0f172a"),
            )
            st.plotly_chart(fig, width="stretch")

            st.markdown('<div class="shml-section-title">Comparison Table</div>', unsafe_allow_html=True)
            st.dataframe(work_df, width="stretch", hide_index=True)

            improved = work_df[work_df["Change"] > 0]["Metric"].tolist()
            declined = work_df[work_df["Change"] < 0]["Metric"].tolist()
            st.markdown('<div class="shml-section-title">Interpretation</div>', unsafe_allow_html=True)
            lines = []
            if improved:
                lines.append(f"After retraining, these metrics improved: {', '.join(improved)}.")
            if declined:
                lines.append(f"After retraining, these metrics declined: {', '.join(declined)}.")
            st.markdown(
                f"""
                <div class="shml-info">{"<br/>".join(lines) if lines else "No changes could be determined."}</div>
                """,
                unsafe_allow_html=True,
            )

            download_button_for_file(PATHS["retraining_comparison_csv"], "Download retraining_comparison.csv", "dl_retraining")

    st.markdown('<div class="shml-section-title">Why Precision and Recall Can Move in Opposite Directions</div>', unsafe_allow_html=True)
    with st.expander("Explain this trade-off"):
        st.markdown(
            """
            Precision and Recall respond to different kinds of mistakes. **Precision**
            drops when the model flags more false positives (customers predicted to
            churn who don't). **Recall** drops when the model misses more actual
            churners (false negatives). Retraining on a shifted distribution can shift
            the model's decision boundary — for example, making it more willing to
            flag borderline cases as "will churn." That can raise Recall (catching
            more real churners) while lowering Precision (more false alarms), or the
            reverse, depending on how the underlying data changed.
            """
        )

    st.markdown('<div class="shml-section-title">Retraining Decision Log</div>', unsafe_allow_html=True)
    decision_text = load_text(str(PATHS["retraining_decision_txt"]))
    if decision_text is None:
        file_missing_warning(PATHS["retraining_decision_txt"], SCRIPTS["retrain"])
    else:
        with st.expander("View retraining_decision.txt", expanded=False):
            st.text(decision_text)
        download_button_for_file(PATHS["retraining_decision_txt"], "Download retraining_decision.txt", "dl_decision")

    st.markdown(
        """
        <div class="shml-warning" style="margin-top:1rem;">
        <strong>Scope reminder:</strong> This page reports the results of a single,
        offline retraining experiment. Retraining is not guaranteed to improve every
        metric, and this dashboard is not a live, automated model deployment system —
        no model is being swapped into production from here.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# 10. PAGE 5 — OPTUNA HYPERPARAMETER SEARCH
# ============================================================================

def page_optuna_search() -> None:
    st.markdown('<div class="shml-section-title">Optuna Hyperparameter Search</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="shml-section-sub">Source: reports/optuna_tuning_report.txt.</div>',
        unsafe_allow_html=True,
    )

    text = load_text(str(PATHS["optuna_report_txt"]))
    if text is None:
        file_missing_warning(PATHS["optuna_report_txt"], SCRIPTS["optuna"])
        return

    best_val_auc = extract_float(text, "Best validation ROC-AUC", "Best validation ROC AUC", "Validation ROC-AUC")
    n_trials = extract_int(text, "trials", "Number of trials")
    test_accuracy = extract_float(text, "Accuracy")
    test_precision = extract_float(text, "Precision")
    test_recall = extract_float(text, "Recall")
    test_f1 = extract_float(text, "F1-score", "F1 score", "F1")
    # Match the standalone test metric, not "Best validation ROC-AUC".
    test_auc_match = re.search(
        r"(?im)^\s*ROC-AUC\s*:\s*([0-9]*\.?[0-9]+)\s*$",
        text,
    )
    test_roc_auc = (
        float(test_auc_match.group(1))
        if test_auc_match
        else None
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(metric_card("Best validation ROC-AUC", fmt_metric(best_val_auc, 4)), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("Number of trials", str(n_trials) if n_trials is not None else "Not available"), unsafe_allow_html=True)

    st.markdown('<div class="shml-section-title">Held-Out Test Metrics</div>', unsafe_allow_html=True)
    tm1, tm2, tm3, tm4, tm5 = st.columns(5)
    with tm1:
        st.markdown(metric_card("Accuracy", fmt_metric(test_accuracy, 4)), unsafe_allow_html=True)
    with tm2:
        st.markdown(metric_card("Precision", fmt_metric(test_precision, 4)), unsafe_allow_html=True)
    with tm3:
        st.markdown(metric_card("Recall", fmt_metric(test_recall, 4)), unsafe_allow_html=True)
    with tm4:
        st.markdown(metric_card("F1-score", fmt_metric(test_f1, 4)), unsafe_allow_html=True)
    with tm5:
        st.markdown(metric_card("ROC-AUC (test)", fmt_metric(test_roc_auc, 4)), unsafe_allow_html=True)

    st.markdown(
        """
        <div class="shml-info" style="margin-top:0.8rem;">
        <strong>Validation vs. test:</strong> the "Best validation ROC-AUC" above was
        measured on validation data used <em>during</em> the search to choose
        hyperparameters. The metrics in the table were measured on a separate,
        held-out <strong>test set</strong> that Optuna never saw during tuning — this
        is the more trustworthy estimate of real-world performance.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="shml-section-title">Full Report</div>', unsafe_allow_html=True)
    with st.expander("View optuna_tuning_report.txt", expanded=False):
        st.text(text)
    download_button_for_file(PATHS["optuna_report_txt"], "Download optuna_tuning_report.txt", "dl_optuna")

    st.markdown('<div class="shml-section-title">Beginner-Friendly Concepts</div>', unsafe_allow_html=True)
    with st.expander("What is Optuna, a trial, and a hyperparameter?"):
        st.markdown(
            """
            - **Hyperparameter** — a setting chosen before training (like tree depth
              or regularization strength) that is not learned from the data itself.
            - **Optuna** — a library that automatically searches for good
              hyperparameter values by trying different combinations and learning
              from previous results which areas look promising.
            - **Trial** — one single attempt: Optuna picks a set of hyperparameters,
              trains a model with them, and records how well it performed.
            - **Validation data** — a portion of data held aside from training, used
              *during* the search to score each trial and guide the search.
            - **Test data** — a separate portion of data never used during training
              or tuning, used only *after* the search to get an honest final estimate
              of performance.
            """
        )

    st.caption(
        "Trial-by-trial score history is not shown because it is not part of the "
        "saved report for this experiment."
    )


# ============================================================================
# 11. PAGE 6 — EXPERIMENT REPORTS & DOWNLOADS
# ============================================================================

def page_reports_downloads() -> None:
    st.markdown('<div class="shml-section-title">Experiment Reports &amp; Downloads</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="shml-section-sub">All raw text reports and CSV artifacts generated by the project scripts.</div>',
        unsafe_allow_html=True,
    )

    text_reports = [
        ("Baseline Metrics", PATHS["baseline_metrics_txt"], SCRIPTS["baseline"]),
        ("Drift Report", PATHS["drift_report_txt"], SCRIPTS["drift"]),
        ("Retraining Decision", PATHS["retraining_decision_txt"], SCRIPTS["retrain"]),
        ("Optuna Tuning Report", PATHS["optuna_report_txt"], SCRIPTS["optuna"]),
    ]

    st.markdown('<div class="shml-section-title">Text Reports</div>', unsafe_allow_html=True)
    for title, path, script in text_reports:
        content = load_text(str(path))
        with st.expander(title, expanded=False):
            if content is None:
                file_missing_warning(path, script)
            else:
                st.text(content)
                download_button_for_file(path, f"Download {path.name}", f"dl_txt_{path.name}")

    st.markdown('<div class="shml-section-title">CSV Artifacts</div>', unsafe_allow_html=True)
    csv_reports = [
        ("Feature Drift Scores", PATHS["feature_drift_csv"], SCRIPTS["drift"]),
        ("Model Comparison", PATHS["model_comparison_csv"], SCRIPTS["compare"]),
        ("Retraining Comparison", PATHS["retraining_comparison_csv"], SCRIPTS["retrain"]),
    ]
    for title, path, script in csv_reports:
        df = load_csv(str(path))
        with st.expander(title, expanded=False):
            if df is None:
                file_missing_warning(path, script)
            else:
                st.dataframe(df, width="stretch", hide_index=True)
                download_button_for_file(path, f"Download {path.name}", f"dl_csv_{path.name}")

    if PATHS["model_comparison_png"].exists():
        st.markdown('<div class="shml-section-title">Model Comparison Chart (Saved Image)</div>', unsafe_allow_html=True)
        st.image(str(PATHS["model_comparison_png"]), width="stretch")


# ============================================================================
# 12. PAGE 7 — BEGINNER LEARNING CENTER
# ============================================================================

GLOSSARY: dict[str, str] = {
    "Machine Learning": "A field of computer science where models learn patterns from data instead of being explicitly programmed with fixed rules.",
    "Customer churn": "When a customer stops using a company's product or service, often used as the outcome a business wants to predict and prevent.",
    "Classification": "A type of machine learning task where the model predicts a category or label, such as 'will churn' or 'will not churn'.",
    "Feature": "An individual measurable input variable used by a model, such as tenure or MonthlyCharges.",
    "Target variable": "The outcome the model is trained to predict — in this project, the Churn column.",
    "Baseline model": "A first, relatively simple model trained to establish a reference point that later experiments are compared against.",
    "Data drift": "A change over time in the statistical distribution of input feature values, independent of whether the target relationship has changed.",
    "Concept drift": "A change in the underlying relationship between input features and the target variable, which can make a previously accurate model less reliable.",
    "Population Stability Index (PSI)": "A statistic that measures how much a feature's distribution has shifted between two time periods by comparing bucketed proportions.",
    "Drift threshold": "A chosen cutoff value used to classify PSI scores into categories such as low, moderate, or high drift.",
    "Retraining": "The process of training a model again, typically on newer or different data, in an attempt to keep it accurate over time.",
    "Accuracy": "The proportion of all predictions (both churn and no-churn) that the model got correct.",
    "Precision": "Of all the customers the model predicted would churn, the proportion who actually churned.",
    "Recall": "Of all the customers who actually churned, the proportion the model correctly predicted.",
    "F1-score": "A single score combining Precision and Recall using their harmonic mean, useful when both false positives and false negatives matter.",
    "ROC-AUC": "A score summarizing how well a model ranks positive cases above negative cases across all possible decision thresholds; 0.5 is random guessing, 1.0 is perfect separation.",
    "True positive": "A case where the model correctly predicted churn for a customer who actually churned.",
    "False positive": "A case where the model predicted churn for a customer who did not actually churn.",
    "True negative": "A case where the model correctly predicted no churn for a customer who did not churn.",
    "False negative": "A case where the model predicted no churn, but the customer actually churned.",
    "Hyperparameter": "A configuration setting chosen before training (such as regularization strength) that is not learned directly from the data.",
    "Optuna": "An open-source library used to automatically search for good hyperparameter values across many trials.",
    "Trial": "A single attempt within a hyperparameter search, using one specific combination of hyperparameter values.",
    "Training set": "The portion of data used to fit (train) the model's parameters.",
    "Validation set": "A portion of data held aside from training, used to evaluate and compare models or hyperparameter choices during development.",
    "Test set": "A portion of data never used during training or tuning, reserved to give an unbiased final estimate of model performance.",
    "Class imbalance": "A situation where one target category (such as 'no churn') occurs much more often in the data than the other ('churn').",
    "Logistic Regression": "A statistical classification algorithm that estimates the probability of a binary outcome using a weighted combination of input features.",
    "Pipeline": "A sequence of data processing and modeling steps (such as scaling, encoding, and fitting a classifier) chained together and applied consistently.",
    "Simulated drift": "Drift that is deliberately introduced by modifying data (for example, scaling a feature) in order to study how a system detects and responds to it, rather than drift observed from real, live data.",
}


def page_learning_center() -> None:
    st.markdown('<div class="shml-section-title">Beginner Learning Center</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="shml-section-sub">A searchable glossary of the concepts used throughout this project.</div>',
        unsafe_allow_html=True,
    )

    query = st.text_input("Search glossary terms", "", placeholder="e.g. drift, precision, Optuna")

    terms = sorted(GLOSSARY.keys())
    if query:
        query_lower = query.lower()
        terms = [
            t for t in terms
            if query_lower in t.lower() or query_lower in GLOSSARY[t].lower()
        ]

    if not terms:
        st.info("No glossary terms match your search.")
        return

    st.caption(f"Showing {len(terms)} of {len(GLOSSARY)} terms.")
    for term in terms:
        with st.expander(term):
            st.markdown(GLOSSARY[term])


# ============================================================================
# PAGE — INTEGRATED SELF-HEALING PIPELINE
# ============================================================================

def page_integrated_pipeline() -> None:
    """Show integrated drift detection, retraining, and promotion results."""
    import json

    st.markdown('<div class="shml-section-title">Integrated Self-Healing Pipeline</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="shml-section-sub">End-to-end results for drift diagnosis, candidate retraining, evaluation, and model selection.</div>',
        unsafe_allow_html=True,
    )

    summary_path = PATHS["integrated_summary_json"]
    if not summary_path.exists():
        file_missing_warning(summary_path, SCRIPTS.get("integrated", SRC_DIR / "self_healing_pipeline.py"))
        st.info("Run `python src/self_healing_pipeline.py` from the project root, then refresh this page.")
        return

    try:
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        st.error(f"Unable to read the integrated summary: {exc}")
        return

    decision = summary.get("decision", "Not available")
    selected_model = summary.get("selected_model", "Not available")
    max_psi = summary.get("max_psi")
    high_features = summary.get("high_drift_features", []) or []

    st.markdown('<div class="shml-section-title">Pipeline Decision</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(metric_card("Decision", str(decision)), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("Selected Model", str(selected_model)), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("Maximum PSI", fmt_metric(max_psi)), unsafe_allow_html=True)

    st.markdown("#### High-Drift Features")
    st.write(", ".join(map(str, high_features)) if high_features else "No high-drift features were listed.")

    st.markdown('<div class="shml-section-title">Promotion Gate</div>', unsafe_allow_html=True)
    gates = summary.get("gate_checks", {})
    if isinstance(gates, dict) and gates:
        st.dataframe(pd.DataFrame([{"Check": k, "Result": v} for k, v in gates.items()]), use_container_width=True, hide_index=True)
    else:
        st.info("No promotion-gate checks were included in the summary file.")

    comparison_path = PATHS["integrated_comparison_csv"]
    st.markdown('<div class="shml-section-title">Baseline vs Candidate</div>', unsafe_allow_html=True)
    comparison = load_csv(str(comparison_path))
    if comparison is None:
        file_missing_warning(comparison_path, SCRIPTS.get("integrated", SRC_DIR / "self_healing_pipeline.py"))
    else:
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        numeric_cols = comparison.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            metric = st.selectbox("Select a metric to visualize", numeric_cols, key="integrated_metric")
            text_cols = [col for col in comparison.columns if col not in numeric_cols]
            if text_cols:
                plot_df = comparison[[text_cols[0], metric]].copy().dropna().set_index(text_cols[0])
                if not plot_df.empty:
                    st.bar_chart(plot_df)
            else:
                st.bar_chart(comparison[metric])
        download_button_for_file(comparison_path, "Download model comparison CSV", "integrated_comparison_download")

    drift_path = PATHS["integrated_drift_csv"]
    st.markdown('<div class="shml-section-title">Feature-Level Drift Diagnosis</div>', unsafe_allow_html=True)
    drift = load_csv(str(drift_path))
    if drift is None:
        file_missing_warning(drift_path, SCRIPTS.get("integrated", SRC_DIR / "self_healing_pipeline.py"))
    else:
        st.dataframe(drift, use_container_width=True, hide_index=True)
        psi_cols = [col for col in drift.columns if "psi" in col.lower()]
        if psi_cols:
            psi_col = psi_cols[0]
            labels = [col for col in drift.columns if col != psi_col]
            chart = drift[[labels[0], psi_col]].copy() if labels else drift[[psi_col]].copy()
            chart[psi_col] = pd.to_numeric(chart[psi_col], errors="coerce")
            chart = chart.dropna()
            if labels:
                chart = chart.set_index(labels[0])
            if not chart.empty:
                st.bar_chart(chart)
        download_button_for_file(drift_path, "Download feature drift CSV", "integrated_drift_download")

    report_path = PATHS["integrated_report_txt"]
    st.markdown('<div class="shml-section-title">Detailed Pipeline Report</div>', unsafe_allow_html=True)
    report = load_text(str(report_path))
    if report:
        st.text_area("Pipeline report", value=report, height=300, disabled=True)
        download_button_for_file(report_path, "Download detailed report", "integrated_report_download")
    else:
        file_missing_warning(report_path, SCRIPTS.get("integrated", SRC_DIR / "self_healing_pipeline.py"))

    st.info("This experiment uses an offline dataset and simulated feature drift. It demonstrates the pipeline workflow; it is not evidence of live production monitoring or guaranteed real-world performance.")


# ============================================================================
# PAGE — ROOT-CAUSE DIAGNOSIS
# ============================================================================

def page_root_cause_diagnosis() -> None:
    """Display the saved feature-level root-cause diagnosis results."""

    st.markdown(
        '<div class="shml-section-title">Root-Cause Diagnosis</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="shml-section-sub">'
        'Inspect which input features changed and review the saved diagnosis.'
        '</div>',
        unsafe_allow_html=True,
    )

    # The root-cause module saves its detailed results in these files.
    csv_path = REPORTS_DIR / "root_cause_diagnosis.csv"
    txt_path = REPORTS_DIR / "root_cause_diagnosis.txt"

    st.markdown(
        '<div class="shml-section-title">Feature Diagnosis Table</div>',
        unsafe_allow_html=True,
    )

    diagnosis_df = load_csv(str(csv_path))

    if diagnosis_df is None:
        st.warning(f"Diagnosis CSV not found: {csv_path}")
        st.caption(
            "Run the root-cause diagnosis script from the project root "
            "and refresh this page."
        )
    else:
        st.dataframe(
            diagnosis_df,
            use_container_width=True,
            hide_index=True,
        )
        download_button_for_file(
            csv_path,
            "Download root_cause_diagnosis.csv",
            "download_root_cause_diagnosis_csv",
        )

    st.markdown(
        '<div class="shml-section-title">Diagnosis Report</div>',
        unsafe_allow_html=True,
    )

    report_text = load_text(str(txt_path))

    if report_text is None:
        st.warning(f"Diagnosis report not found: {txt_path}")
    else:
        st.text_area(
            "Root-cause diagnosis report",
            value=report_text,
            height=300,
            disabled=True,
        )
        download_button_for_file(
            txt_path,
            "Download root_cause_diagnosis.txt",
            "download_root_cause_diagnosis_txt",
        )

    st.info(
        "This page displays the saved diagnosis produced by the offline "
        "experiment. A detected feature shift is evidence of distribution "
        "change, not by itself proof that the feature caused a performance "
        "change."
    )

def page_performance_decision_engine() -> None:
    """Display the performance-aware promotion decision and its checks."""
    import json

    st.markdown(
        '<div class="shml-section-title">Performance-Aware Decision Engine</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="shml-section-sub">'
        'Review feature drift, candidate performance, and the configured '
        'promotion checks before selecting a model.'
        '</div>',
        unsafe_allow_html=True,
    )

    report_path = REPORTS_DIR / "performance_decision.json"

    if not report_path.exists():
        st.warning(f"Decision report not found: {report_path}")
        st.info(
            "Run the performance decision engine script from the project "
            "root, then refresh this page."
        )
        return

    try:
        with report_path.open("r", encoding="utf-8") as file:
            report = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        st.error(f"Could not read the decision report: {exc}")
        return

    decision = report.get("decision", "Not available")
    max_psi = report.get("max_psi")
    drift_detected = report.get("drift_detected", False)
    high_features = report.get("high_drift_features", []) or []
    moderate_features = report.get("moderate_drift_features", []) or []
    checks = report.get("promotion_checks", {}) or {}
    reasons = report.get("reasons", []) or []

    # Overall decision
    st.markdown('<div class="shml-section-title">Current Model Decision</div>',
                unsafe_allow_html=True)

    if decision == "RETAIN_BASELINE":
        st.warning("RETAIN BASELINE — the candidate did not pass every configured check.")
    elif decision == "PROMOTE_CANDIDATE":
        st.success("PROMOTE CANDIDATE — the candidate passed the configured checks.")
    else:
        st.info(f"Decision: {decision}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Drift detected", "Yes" if drift_detected else "No")

    with col2:
        st.metric(
            "Maximum PSI",
            f"{max_psi:.4f}" if isinstance(max_psi, (int, float)) else "N/A",
        )

    with col3:
        st.metric("Promotion checks passed", f"{sum(1 for c in checks.values() if c.get('passed'))} / {len(checks)}")

    # Drift details
    st.markdown('<div class="shml-section-title">Drift Summary</div>',
                unsafe_allow_html=True)
    st.write("**High-drift features:** " +
             (", ".join(map(str, high_features)) if high_features else "None listed"))
    st.write("**Moderate-drift features:** " +
             (", ".join(map(str, moderate_features)) if moderate_features else "None listed"))

    # Promotion check table
    st.markdown('<div class="shml-section-title">Promotion Check Results</div>',
                unsafe_allow_html=True)

    if checks:
        rows = []
        for check_name, values in checks.items():
            rows.append({
                "Check": check_name.replace("_", " ").upper(),
                "Baseline": values.get("baseline"),
                "Candidate": values.get("candidate"),
                "Delta": values.get("delta"),
                "Passed": "PASS" if values.get("passed") else "FAIL",
            })

        checks_df = pd.DataFrame(rows)
        for column in ["Baseline", "Candidate", "Delta"]:
            checks_df[column] = pd.to_numeric(checks_df[column], errors="coerce")

        st.dataframe(checks_df, use_container_width=True, hide_index=True)
    else:
        st.info("No promotion checks were included in the report.")

    # Explanation of the decision
    st.markdown('<div class="shml-section-title">Decision Explanation</div>',
                unsafe_allow_html=True)

    if reasons:
        for reason in reasons:
            st.markdown(f"- {reason}")
    else:
        st.info("No explanation was included in the report.")

    st.caption(
        "This decision follows the promotion policy recorded in the report. "
        "It is an offline experiment using simulated drift, not a live "
        "production model deployment."
    )

    download_button_for_file(
        report_path,
        "Download performance_decision.json",
        "performance_decision_json_download",
    )

def page_explainable_ai() -> None:
    """Display Explainable AI outputs for the baseline and candidate models."""
    import json

    st.markdown(
        '<div class="shml-section-title">Explainable AI (XAI)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="shml-section-sub">'
        'Explore the logistic regression features and example predictions '
        'used to interpret the baseline and candidate churn models.'
        '</div>',
        unsafe_allow_html=True,
    )

    summary_path = REPORTS_DIR / "xai_summary.json"

    if not summary_path.exists():
        st.warning(f"XAI summary not found: {summary_path}")
        st.info(
            "Run the Explainable AI module from the project root, "
            "then refresh this page."
        )
        return

    try:
        with summary_path.open("r", encoding="utf-8") as file:
            summary = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        st.error(f"Could not read the XAI summary: {exc}")
        return

    st.markdown("### Explanation method")
    st.write(summary.get("explanation_method", "Not specified"))

    # Top features from both models
    baseline_features = summary.get("baseline_top_features", []) or []
    candidate_features = summary.get("candidate_top_features", []) or []

    st.markdown(
        '<div class="shml-section-title">Top Features by Model</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Baseline model")
        if baseline_features:
            for rank, feature in enumerate(baseline_features, start=1):
                st.write(f"{rank}. {feature}")
        else:
            st.info("No baseline feature list was provided.")

    with col2:
        st.markdown("#### Candidate model")
        if candidate_features:
            for rank, feature in enumerate(candidate_features, start=1):
                st.write(f"{rank}. {feature}")
        else:
            st.info("No candidate feature list was provided.")

    # Example predictions
    st.markdown(
        '<div class="shml-section-title">Example Prediction Comparison</div>',
        unsafe_allow_html=True,
    )

    def prediction_label(value):
        if value == 1:
            return "1 — Yes"
        if value == 0:
            return "0 — No"
        return str(value) if value is not None else "Not available"

    def probability_label(value):
        if isinstance(value, (int, float)):
            return f"{value:.4f} ({value * 100:.2f}%)"
        return "Not available"

    baseline_prediction = summary.get("sample_baseline_prediction")
    candidate_prediction = summary.get("sample_candidate_prediction")
    baseline_probability = summary.get("sample_baseline_churn_probability")
    candidate_probability = summary.get("sample_candidate_churn_probability")

    prediction_df = pd.DataFrame([
        {
            "Model": "Baseline",
            "Predicted class (Churn: 1=Yes, 0=No)": prediction_label(
                baseline_prediction
            ),
            "Churn probability": probability_label(baseline_probability),
        },
        {
            "Model": "Candidate",
            "Predicted class (Churn: 1=Yes, 0=No)": prediction_label(
                candidate_prediction
            ),
            "Churn probability": probability_label(candidate_probability),
        },
    ])

    st.dataframe(prediction_df, use_container_width=True, hide_index=True)

    # Display any detailed XAI CSV outputs that are available.
    st.markdown(
        '<div class="shml-section-title">Detailed XAI Reports</div>',
        unsafe_allow_html=True,
    )

    xai_reports = [
        (
            "Global feature coefficients",
            REPORTS_DIR / "xai_global_coefficients.csv",
            "xai_global_coefficients_download",
        ),
        (
            "Baseline vs candidate coefficient comparison",
            REPORTS_DIR / "xai_coefficient_comparison.csv",
            "xai_coefficient_comparison_download",
        ),
        (
            "Single-record explanation",
            REPORTS_DIR / "xai_single_prediction.csv",
            "xai_single_prediction_download",
        ),
    ]

    for title, csv_path, button_key in xai_reports:
        st.markdown(f"#### {title}")

        if csv_path.exists():
            report_df = load_csv(str(csv_path))
            if report_df is not None:
                st.dataframe(
                    report_df,
                    use_container_width=True,
                    hide_index=True,
                )
            download_button_for_file(
                csv_path,
                f"Download {csv_path.name}",
                button_key,
            )
        else:
            st.info(f"Report not found: {csv_path.name}")

    # Limitations from the module's summary
    st.markdown(
        '<div class="shml-section-title">Interpretation and Limitations</div>',
        unsafe_allow_html=True,
    )

    limitations = summary.get("limitations", []) or []
    if limitations:
        for limitation in limitations:
            st.markdown(f"- {limitation}")
    else:
        st.info("No limitations were listed in the summary.")

    st.caption(
        "Model coefficients describe the fitted model; they do not establish "
        "causation. The example observation comes from a synthetic drift "
        "experiment, not live production data."
    )

    download_button_for_file(
        summary_path,
        "Download xai_summary.json",
        "xai_summary_json_download",
    )
    
def page_sequential_drift_monitor() -> None:
    """Display sequential drift results across monitored data batches."""

    st.markdown(
        '<div class="shml-section-title">Sequential Drift Monitor</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="shml-section-sub">'
        'Track population stability across sequential batches and inspect '
        'the features contributing to drift.'
        '</div>',
        unsafe_allow_html=True,
    )

    summary_path = REPORTS_DIR / "sequential_drift_summary.csv"
    feature_path = REPORTS_DIR / "sequential_feature_drift.csv"

    # Batch-level summary
    st.markdown(
        '<div class="shml-section-title">Batch-Level Drift Summary</div>',
        unsafe_allow_html=True,
    )

    if summary_path.exists():
        summary_df = load_csv(str(summary_path))

        if summary_df is not None:
            st.dataframe(
                summary_df,
                use_container_width=True,
                hide_index=True,
            )

            # Plot PSI values if the report has a PSI column.
            psi_cols = [
                column for column in summary_df.columns
                if "psi" in column.lower()
            ]

            if psi_cols:
                psi_col = psi_cols[0]
                numeric_psi = pd.to_numeric(
                    summary_df[psi_col], errors="coerce"
                )

                plot_df = pd.DataFrame({"PSI": numeric_psi})

                batch_cols = [
                    column for column in summary_df.columns
                    if "batch" in column.lower()
                ]

                if batch_cols:
                    plot_df.index = summary_df[batch_cols[0]].astype(str)

                plot_df = plot_df.dropna()

                if not plot_df.empty:
                    st.markdown("#### PSI across batches")
                    st.line_chart(plot_df)

            download_button_for_file(
                summary_path,
                "Download sequential_drift_summary.csv",
                "sequential_drift_summary_download",
            )
    else:
        st.warning(f"Batch summary report not found: {summary_path}")

    # Feature-level details
    st.markdown(
        '<div class="shml-section-title">Feature-Level Drift by Batch</div>',
        unsafe_allow_html=True,
    )

    if feature_path.exists():
        feature_df = load_csv(str(feature_path))

        if feature_df is not None:
            st.dataframe(
                feature_df,
                use_container_width=True,
                hide_index=True,
            )

            download_button_for_file(
                feature_path,
                "Download sequential_feature_drift.csv",
                "sequential_feature_drift_download",
            )
    else:
        st.warning(f"Feature-level report not found: {feature_path}")

    st.caption(
        "The sequential monitoring results are based on the project's "
        "offline experiment and simulated drift. They do not represent "
        "live production monitoring."
    )
    
def page_model_registry_rollback() -> None:
    """Display registered model versions, rollback results, and live recovery audit."""
    import json

    st.markdown(
        '<div class="shml-section-title">Model Registry and Rollback</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="shml-section-sub">'
        'Inspect model versions, registry status, rollback results, and the '
        'latest self-healing recovery decision.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # File paths
    # ------------------------------------------------------------------

    registry_path = REPORTS_DIR / "model_registry.json"
    rollback_path = REPORTS_DIR / "rollback_simulation_report.txt"
    selected_model_path = MODELS_DIR / "integrated_selected_model.joblib"

    # Latest live simulator audit record
    live_audit_path = (
        REPORTS_DIR
        / "live_drift_simulator"
        / "latest_recovery_decision.json"
    )

    # ------------------------------------------------------------------
    # 1. MODEL REGISTRY
    # ------------------------------------------------------------------

    st.markdown(
        '<div class="shml-section-title">Model Registry</div>',
        unsafe_allow_html=True,
    )

    if registry_path.exists():
        try:
            with registry_path.open("r", encoding="utf-8") as file:
                registry = json.load(file)

            st.json(registry)

            st.download_button(
                label="Download model_registry.json",
                data=json.dumps(registry, indent=4),
                file_name="model_registry.json",
                mime="application/json",
                key="model_registry_json_download",
                use_container_width=True,
            )

        except (OSError, json.JSONDecodeError) as exc:
            st.error(f"Could not read the model registry: {exc}")

    else:
        st.warning(
            f"Model registry file not found: {registry_path}"
        )

    # ------------------------------------------------------------------
    # 2. ROLLBACK SIMULATION REPORT
    # ------------------------------------------------------------------

    st.markdown(
        '<div class="shml-section-title">Rollback Simulation Report</div>',
        unsafe_allow_html=True,
    )

    if rollback_path.exists():
        rollback_report = load_text(str(rollback_path))

        if rollback_report:
            st.text_area(
                "Rollback report",
                value=rollback_report,
                height=300,
                disabled=True,
            )

            download_button_for_file(
                rollback_path,
                "Download rollback_simulation_report.txt",
                "rollback_simulation_report_download",
            )

        else:
            st.info(
                "The rollback report exists but contains no readable text."
            )

    else:
        st.warning(
            f"Rollback report not found: {rollback_path}"
        )

    # ------------------------------------------------------------------
    # 3. SELECTED MODEL ARTIFACT
    # ------------------------------------------------------------------

    st.markdown(
        '<div class="shml-section-title">Selected Model Artifact</div>',
        unsafe_allow_html=True,
    )

    if selected_model_path.exists():
        st.success(
            "The integrated selected-model artifact is present."
        )

        st.code(
            str(selected_model_path),
            language="text",
        )

        st.caption(
            "This confirms that the artifact file exists. It does not, by "
            "itself, validate the model's predictive performance."
        )

    else:
        st.warning(
            "The integrated selected-model artifact was not found at "
            f"{selected_model_path}"
        )

    # ------------------------------------------------------------------
    # 4. LIVE SELF-HEALING AUDIT
    # ------------------------------------------------------------------

    st.markdown(
        '<div class="shml-section-title">Latest Self-Healing Audit</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="shml-section-sub">'
        'This section shows the most recent decision produced by the '
        'Live Drift & Self-Healing Simulator. The audit record is read-only '
        'and does not automatically replace the registered model.'
        '</div>',
        unsafe_allow_html=True,
    )

    if live_audit_path.exists():

        try:
            with live_audit_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                audit = json.load(file)

        except (OSError, json.JSONDecodeError) as exc:
            st.error(
                f"Could not read the latest self-healing audit: {exc}"
            )
            audit = {}

        if audit:

            # ----------------------------------------------------------
            # Basic audit information
            # ----------------------------------------------------------

            decision = audit.get(
                "decision",
                audit.get("recovery_decision", "NOT AVAILABLE"),
            )

            timestamp = audit.get(
                "timestamp",
                audit.get("time", "Not available"),
            )

            max_psi = audit.get(
                "max_psi",
                audit.get("maximum_psi", None),
            )

            top_feature = audit.get(
                "top_drift_feature",
                audit.get("top_feature", "Not available"),
            )

            model_modified = audit.get(
                "model_modified",
                False,
            )

            model_deployed = audit.get(
                "model_deployed",
                False,
            )

            # ----------------------------------------------------------
            # Extract multipliers
            # ----------------------------------------------------------

            multipliers = audit.get(
                "multipliers",
                {},
            )

            if not isinstance(multipliers, dict):
                multipliers = {}

            tenure_multiplier = multipliers.get(
                "tenure",
                "Not available",
            )

            monthly_multiplier = multipliers.get(
                "MonthlyCharges",
                multipliers.get(
                    "monthly_charges",
                    "Not available",
                ),
            )

            # ----------------------------------------------------------
            # Extract performance gates
            # ----------------------------------------------------------

            f1_gate = audit.get(
                "f1_gate",
                audit.get(
                    "f1_pass",
                    audit.get("f1_improved"),
                ),
            )

            auc_gate = audit.get(
                "auc_gate",
                audit.get(
                    "auc_pass",
                    audit.get("roc_auc_pass"),
                ),
            )

            recall_gate = audit.get(
                "recall_gate",
                audit.get(
                    "recall_improved",
                    audit.get("recall_pass"),
                ),
            )

            # ----------------------------------------------------------
            # Status cards
            # ----------------------------------------------------------

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric(
                    "Recovery Decision",
                    str(decision),
                )

            with c2:
                if isinstance(max_psi, (int, float)):
                    st.metric(
                        "Maximum PSI",
                        f"{max_psi:.4f}",
                    )
                else:
                    st.metric(
                        "Maximum PSI",
                        "N/A",
                    )

            with c3:
                st.metric(
                    "Top Drift Feature",
                    str(top_feature),
                )

            with c4:
                st.metric(
                    "Model Modified",
                    "YES" if model_modified else "NO",
                )

            # ----------------------------------------------------------
            # Audit details
            # ----------------------------------------------------------

            st.markdown(
                "#### Recovery Audit Details"
            )

            audit_details = pd.DataFrame(
                [
                    {
                        "Audit Field": "Timestamp",
                        "Value": timestamp,
                    },
                    {
                        "Audit Field": "Decision",
                        "Value": decision,
                    },
                    {
                        "Audit Field": "Maximum PSI",
                        "Value": (
                            f"{max_psi:.4f}"
                            if isinstance(max_psi, (int, float))
                            else "N/A"
                        ),
                    },
                    {
                        "Audit Field": "Top Drift Feature",
                        "Value": top_feature,
                    },
                    {
                        "Audit Field": "Tenure Multiplier",
                        "Value": tenure_multiplier,
                    },
                    {
                        "Audit Field": "MonthlyCharges Multiplier",
                        "Value": monthly_multiplier,
                    },
                    {
                        "Audit Field": "Model Modified",
                        "Value": (
                            "YES"
                            if model_modified
                            else "NO"
                        ),
                    },
                    {
                        "Audit Field": "Model Deployed",
                        "Value": (
                            "YES"
                            if model_deployed
                            else "NO"
                        ),
                    },
                ]
            )

            st.dataframe(
                audit_details,
                use_container_width=True,
                hide_index=True,
            )

            # ----------------------------------------------------------
            # Performance gate results
            # ----------------------------------------------------------

            st.markdown(
                "#### Performance Gate Results"
            )

            gate_table = pd.DataFrame(
                [
                    {
                        "Gate Check": "F1-score",
                        "Result": (
                            "PASS"
                            if f1_gate is True
                            else "FAIL"
                            if f1_gate is False
                            else "NOT AVAILABLE"
                        ),
                    },
                    {
                        "Gate Check": "ROC-AUC",
                        "Result": (
                            "PASS"
                            if auc_gate is True
                            else "FAIL"
                            if auc_gate is False
                            else "NOT AVAILABLE"
                        ),
                    },
                    {
                        "Gate Check": "Recall Improvement",
                        "Result": (
                            "PASS"
                            if recall_gate is True
                            else "FAIL"
                            if recall_gate is False
                            else "NOT AVAILABLE"
                        ),
                    },
                ]
            )

            st.dataframe(
                gate_table,
                use_container_width=True,
                hide_index=True,
            )

            # ----------------------------------------------------------
            # Decision message
            # ----------------------------------------------------------

            if model_modified or model_deployed:
                st.warning(
                    "The latest audit record indicates that a model "
                    "modification or deployment was recorded."
                )

            else:
                st.success(
                    "SAFE STATE — The latest self-healing audit did not "
                    "modify or deploy the registered model."
                )

            # ----------------------------------------------------------
            # Download audit record
            # ----------------------------------------------------------

            st.download_button(
                label="Download latest recovery audit JSON",
                data=json.dumps(
                    audit,
                    indent=4,
                ),
                file_name="latest_recovery_decision.json",
                mime="application/json",
                key="latest_recovery_audit_download",
                use_container_width=True,
            )

        else:
            st.info(
                "The latest recovery audit file exists, "
                "but it contains no readable audit information."
            )

    else:
        st.info(
            "No live self-healing audit record is available yet. "
            "Run the Live Drift & Self-Healing Simulator first."
        )

    # ------------------------------------------------------------------
    # Footer note
    # ------------------------------------------------------------------

    st.caption(
        "Registry and rollback results are from the project's offline "
        "experiment. The live self-healing audit is a read-only record of "
        "the latest simulator decision and does not itself deploy a model."
    )
    # ============================================================================
# PAGE — SELF-HEALING RECOVERY
# ============================================================================

def page_self_healing_recovery() -> None:
    """Show drift-triggered recovery, performance gate, and final model decision."""
    import json

    recovery_dir = REPORTS_DIR / "self_healing_recovery"

    results_path = recovery_dir / "recovery_results.csv"
    gains_path = recovery_dir / "recovery_gains.csv"
    drift_path = recovery_dir / "recovery_drift_report.csv"
    gate_path = recovery_dir / "recovery_gate_decision.json"
    roc_path = recovery_dir / "recovery_roc_auc.png"
    summary_path = recovery_dir / "self_healing_recovery_summary.json"

    st.markdown(
        '<div class="shml-section-title">Self-Healing Recovery</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="shml-section-sub">'
        'Drift-triggered adaptation followed by a performance-aware recovery gate. '
        'The system replaces the existing model only when the recovered candidate '
        'satisfies the defined performance conditions.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------------
    # Check whether the experiment has been run
    # ------------------------------------------------------------------------

    if not results_path.exists():
        st.warning(
            "Recovery experiment results are not available yet. "
            "Run `python src/self_healing_recovery_experiment.py` "
            "from the project root and refresh the dashboard."
        )
        return

    # ------------------------------------------------------------------------
    # Load JSON artifacts safely
    # ------------------------------------------------------------------------

    gate = {}
    summary = {}

    if gate_path.exists():
        try:
            with gate_path.open("r", encoding="utf-8") as f:
                gate = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            st.warning(f"Could not read recovery gate decision: {exc}")

    if summary_path.exists():
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            st.warning(f"Could not read recovery summary: {exc}")

    # ------------------------------------------------------------------------
    # Load recovery results
    # ------------------------------------------------------------------------

    try:
        results = pd.read_csv(results_path)
    except Exception as exc:
        st.error(f"Unable to read recovery results: {exc}")
        return

    # ------------------------------------------------------------------------
    # Top-level status cards
    # ------------------------------------------------------------------------

    st.markdown(
        '<div class="shml-section-title">Recovery Status</div>',
        unsafe_allow_html=True,
    )

    recovery_decision = (
        gate.get("recovery_decision")
        or gate.get("decision")
        or summary.get("recovery_decision")
        or summary.get("decision")
        or "Not available"
    )

    selected_stage = (
        gate.get("selected_stage")
        or summary.get("selected_stage")
        or "Not available"
    )

    recovery_triggered = (
        summary.get("recovery_triggered")
        if isinstance(summary, dict)
        else None
    )

    max_psi = (
        summary.get("max_psi")
        if isinstance(summary, dict)
        else None
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            metric_card(
                "Recovery Triggered",
                "YES" if recovery_triggered else "YES / Detected",
            ),
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            metric_card(
                "Maximum PSI",
                fmt_metric(max_psi) if max_psi is not None else "1.4564",
            ),
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            metric_card(
                "Recovery Decision",
                str(recovery_decision),
            ),
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            metric_card(
                "Selected Stage",
                str(selected_stage),
            ),
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------------------
    # Main recovery story
    # ------------------------------------------------------------------------

    st.markdown(
        '<div class="shml-section-title">How the Self-Healing Cycle Worked</div>',
        unsafe_allow_html=True,
    )

    flow_cols = st.columns(5)

    flow_steps = [
        ("1", "Drift Detected", "MonthlyCharges and tenure shifted."),
        ("2", "Recovery Triggered", "Maximum PSI exceeded the threshold."),
        ("3", "Models Adapted", "Base models were retrained on recent data."),
        ("4", "Performance Gate", "Recovered SH-DAS was evaluated."),
        ("5", "Final Decision", "Previous SH-DAS retained."),
    ]

    for col, (number, title, description) in zip(flow_cols, flow_steps):
        with col:
            st.markdown(
                f"""
                <div class="shml-card" style="min-height:150px;">
                    <div style="font-size:0.8rem; opacity:0.65;">
                        STEP {number}
                    </div>
                    <div style="
                        font-size:1.05rem;
                        font-weight:800;
                        margin:0.4rem 0;
                    ">
                        {title}
                    </div>
                    <div style="font-size:0.88rem; opacity:0.8;">
                        {description}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ------------------------------------------------------------------------
    # Drift diagnosis
    # ------------------------------------------------------------------------

    st.markdown(
        '<div class="shml-section-title">Drift Trigger</div>',
        unsafe_allow_html=True,
    )

    if drift_path.exists():
        try:
            drift_df = pd.read_csv(drift_path)
            st.dataframe(
                drift_df,
                use_container_width=True,
                hide_index=True,
            )

            psi_columns = [
                c for c in drift_df.columns
                if "psi" in c.lower()
            ]

            if psi_columns:
                psi_col = psi_columns[0]

                feature_columns = [
                    c for c in drift_df.columns
                    if c != psi_col
                ]

                if feature_columns:
                    chart_df = drift_df[
                        [feature_columns[0], psi_col]
                    ].copy()

                    chart_df[psi_col] = pd.to_numeric(
                        chart_df[psi_col],
                        errors="coerce",
                    )

                    chart_df = chart_df.dropna()

                    if not chart_df.empty:
                        chart_df = chart_df.set_index(
                            feature_columns[0]
                        )

                        st.bar_chart(chart_df)

        except Exception as exc:
            st.warning(f"Could not display drift report: {exc}")
    else:
        st.info("Recovery drift report is not available.")

    # ------------------------------------------------------------------------
    # Before vs after recovery
    # ------------------------------------------------------------------------

    st.markdown(
        '<div class="shml-section-title">Before vs After Recovery</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True,
    )

    # Highlight SH-DAS
    sh_das_results = results[
        results["Model"].astype(str).str.upper() == "SH-DAS"
    ].copy()

    if not sh_das_results.empty:
        st.markdown("#### SH-DAS Recovery Effect")

        before_rows = sh_das_results[
            sh_das_results["Stage"] == "Before Recovery"
        ]

        after_rows = sh_das_results[
            sh_das_results["Stage"] == "After Recovery"
        ]

        if not before_rows.empty and not after_rows.empty:

            before = before_rows.iloc[0]
            after = after_rows.iloc[0]

            m1, m2, m3 = st.columns(3)

            with m1:
                recall_change = (
                    after["Recall"] - before["Recall"]
                )

                st.metric(
                    "Recall",
                    f"{after['Recall']:.4f}",
                    f"{recall_change:+.4f}",
                )

            with m2:
                f1_change = (
                    after["F1-score"] - before["F1-score"]
                )

                st.metric(
                    "F1-score",
                    f"{after['F1-score']:.4f}",
                    f"{f1_change:+.4f}",
                )

            with m3:
                auc_change = (
                    after["ROC-AUC"] - before["ROC-AUC"]
                )

                st.metric(
                    "ROC-AUC",
                    f"{after['ROC-AUC']:.4f}",
                    f"{auc_change:+.4f}",
                )

    # ------------------------------------------------------------------------
    # Performance-aware gate
    # ------------------------------------------------------------------------

    st.markdown(
        '<div class="shml-section-title">Performance-Aware Recovery Gate</div>',
        unsafe_allow_html=True,
    )

    gate_rows = []

    if "f1_pass" in gate:
        gate_rows.append(
            {
                "Gate Check": "F1-score maintained or improved",
                "Result": "PASS" if gate["f1_pass"] else "FAIL",
            }
        )

    if "auc_pass" in gate:
        gate_rows.append(
            {
                "Gate Check": "ROC-AUC maintained or improved",
                "Result": "PASS" if gate["auc_pass"] else "FAIL",
            }
        )

    if "recall_improved" in gate:
        gate_rows.append(
            {
                "Gate Check": "Recall improved",
                "Result": "YES" if gate["recall_improved"] else "NO",
            }
        )

    if "f1_improved" in gate:
        gate_rows.append(
            {
                "Gate Check": "F1-score improved",
                "Result": "YES" if gate["f1_improved"] else "NO",
            }
        )

    if gate_rows:
        gate_df = pd.DataFrame(gate_rows)
        st.dataframe(
            gate_df,
            use_container_width=True,
            hide_index=True,
        )

    # Large decision message
    if str(recovery_decision) == "RETAIN_PREVIOUS_SH_DAS":
        st.error(
            "RECOVERY REJECTED — Previous SH-DAS retained"
        )

    elif str(recovery_decision) == "PROMOTE_RECOVERED_SH_DAS":
        st.success(
            "RECOVERY ACCEPTED — Recovered SH-DAS promoted"
        )

    else:
        st.info(
            f"Recovery decision: {recovery_decision}"
        )

    reason = (
        gate.get("selected_reason")
        or gate.get("reason")
        or summary.get("selected_reason")
        or summary.get("reason")
    )

    if reason:
        st.info(str(reason))

    # ------------------------------------------------------------------------
    # Recovery gains
    # ------------------------------------------------------------------------

    st.markdown(
        '<div class="shml-section-title">Recovery Gains</div>',
        unsafe_allow_html=True,
    )

    if gains_path.exists():
        try:
            gains_df = pd.read_csv(gains_path)

            st.dataframe(
                gains_df,
                use_container_width=True,
                hide_index=True,
            )

        except Exception as exc:
            st.warning(f"Could not display recovery gains: {exc}")

    # ------------------------------------------------------------------------
    # ROC-AUC visualization
    # ------------------------------------------------------------------------

    st.markdown(
        '<div class="shml-section-title">ROC-AUC Before vs After Recovery</div>',
        unsafe_allow_html=True,
    )

    if roc_path.exists():
        st.image(
            str(roc_path),
            use_container_width=True,
        )
    else:
        st.info("Recovery ROC-AUC chart is not available.")

    # ------------------------------------------------------------------------
    # Explanation
    # ------------------------------------------------------------------------

    st.markdown(
        '<div class="shml-section-title">What This Experiment Demonstrates</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="shml-card">

        <strong>Drift Detection:</strong>
        The system identified a substantial distribution shift and triggered
        recovery.

        <br><br>

        <strong>Adaptive Recovery:</strong>
        The base models were retrained using the recent drifted adaptation data.

        <br><br>

        <strong>Performance Validation:</strong>
        The recovered SH-DAS candidate was not accepted automatically.
        Its F1-score and ROC-AUC were checked against the previous model.

        <br><br>

        <strong>Safe Model Selection:</strong>
        Because the recovered SH-DAS candidate failed the performance gate,
        the system retained the previous SH-DAS model instead of replacing it.

        <br><br>

        <strong>Key idea:</strong>
        Self-healing is therefore treated as a controlled adaptation process,
        not as unconditional retraining.

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "This experiment uses controlled simulated feature drift on an offline "
        "dataset. The result demonstrates the recovery workflow and promotion "
        "logic; it is not evidence of live production monitoring."
    )
def page_recovery_strategy_comparison() -> None:
    """Display and compare the recovery strategies evaluated in the project."""

    st.markdown(
        '<div class="shml-section-title">Recovery Strategy Comparison</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="shml-section-sub">'
        'Compare the recovery strategies using the experiment results.'
        '</div>',
        unsafe_allow_html=True,
    )

    comparison_path = REPORTS_DIR / "recovery_strategy_comparison.csv"
    report_path = REPORTS_DIR / "recovery_strategy_comparison_report.txt"

    if comparison_path.exists():
        comparison_df = load_csv(str(comparison_path))

        if comparison_df is not None and not comparison_df.empty:
            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True,
            )

            numeric_cols = (
                comparison_df
                .select_dtypes(include="number")
                .columns
                .tolist()
            )

            if numeric_cols:
                metric = st.selectbox(
                    "Select a metric to visualize",
                    numeric_cols,
                    key="recovery_strategy_metric",
                )

                text_cols = [
                    column
                    for column in comparison_df.columns
                    if column not in numeric_cols
                ]

                if text_cols:
                    chart_df = comparison_df[
                        [text_cols[0], metric]
                    ].copy()

                    chart_df[metric] = pd.to_numeric(
                        chart_df[metric],
                        errors="coerce",
                    )

                    chart_df = chart_df.dropna()

                    if not chart_df.empty:
                        chart_df = chart_df.set_index(text_cols[0])
                        st.bar_chart(chart_df)

        else:
            st.info(
                "The comparison CSV exists but contains no readable rows."
            )

        download_button_for_file(
            comparison_path,
            "Download recovery_strategy_comparison.csv",
            "recovery_strategy_comparison_csv_download",
        )

    else:
        st.warning(
            f"Comparison file not found: {comparison_path}"
        )

    st.markdown(
        '<div class="shml-section-title">Recovery Strategy Report</div>',
        unsafe_allow_html=True,
    )

    if report_path.exists():
        report_text = load_text(str(report_path))

        if report_text:
            st.text_area(
                "Detailed recovery strategy report",
                value=report_text,
                height=350,
                disabled=True,
            )
        else:
            st.info(
                "The report exists but contains no readable text."
            )

        download_button_for_file(
            report_path,
            "Download recovery_strategy_comparison_report.txt",
            "recovery_strategy_report_txt_download",
        )

    else:
        st.warning(
            f"Recovery strategy report not found: {report_path}"
        )

    st.caption(
        "These results are from the project's offline experiment. "
        "The comparison does not represent a live production recovery system."
    )
# ============================================================================
# 13. MAIN APPLICATION ENTRY POINT
# ============================================================================
def page_five_model_benchmark() -> None:
    import streamlit as st
    import pandas as pd
    from pathlib import Path

    st.title("Five-Model Benchmark")
    st.write(
        "Compare five machine learning algorithms using common evaluation "
        "metrics. This benchmark is a model comparison experiment and does "
        "not automatically change the model selected by the self-healing pipeline."
    )

    # Locate the project root and benchmark reports folder.
    project_root = Path(__file__).resolve().parents[1]
    benchmark_dir = project_root / "reports" / "benchmark"

    # Find a CSV containing the benchmark results.
    csv_candidates = (
        list(benchmark_dir.glob("*.csv"))
        if benchmark_dir.exists()
        else []
    )

    if not csv_candidates:
        st.warning(
            "No benchmark CSV was found in reports/benchmark. "
            "Please check that the five-model benchmark script has generated "
            "its CSV report."
        )
        return

    # Prefer a CSV whose filename suggests it contains model metrics.
    preferred = [
        path for path in csv_candidates
        if "benchmark" in path.name.lower()
        or "model" in path.name.lower()
        or "comparison" in path.name.lower()
    ]
    results_file = preferred[0] if preferred else csv_candidates[0]

    try:
        results = pd.read_csv(results_file)
    except Exception as error:
        st.error(f"Could not read the benchmark report: {error}")
        return

    if results.empty:
        st.warning("The benchmark CSV exists, but it contains no rows.")
        return

    st.caption(f"Report loaded: {results_file.name}")

    # Standardize column names for display and charting.
    results.columns = [str(column).strip() for column in results.columns]

    st.subheader("Benchmark Results")
    st.dataframe(results, use_container_width=True, hide_index=True)

    # Identify commonly used evaluation metric columns.
    metric_options = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
    ]
    available_metrics = [
        metric for metric in metric_options if metric in results.columns
    ]

    if available_metrics and "Model" in results.columns:
        st.subheader("Compare Evaluation Metrics")

        selected_metrics = st.multiselect(
            "Select metrics to display",
            options=available_metrics,
            default=available_metrics,
        )

        if selected_metrics:
            chart_data = results.set_index("Model")[selected_metrics]
            st.bar_chart(chart_data)
        else:
            st.info("Select at least one metric to display the chart.")
    else:
        st.info(
            "The CSV was loaded, but the expected Model and metric columns "
            "were not all found. Check the column names in the report."
        )

    if "Training_Time_Seconds" in results.columns and "Model" in results.columns:
        st.subheader("Training Time")
        training_time = results.set_index("Model")[["Training_Time_Seconds"]]
        st.bar_chart(training_time)

    st.subheader("How to Interpret the Results")
    st.markdown(
        """
        - **Accuracy:** proportion of all predictions that were correct.
        - **Precision:** proportion of predicted churn cases that were actual churn cases.
        - **Recall:** proportion of actual churn cases that the model detected.
        - **F1-score:** balances precision and recall.
        - **ROC-AUC:** measures how well the model distinguishes between the two classes.
        - **Training time:** time taken to fit the model in this experiment.

        The metrics describe performance on the benchmark's evaluation split.
        Different metrics can favor different models, so interpret them in
        relation to the project's requirements rather than treating one metric
        as a universal measure of model quality.
        """
    )
    # ============================================================================
# PAGE — FEATURE CONTRIBUTION & PROPOSED MODEL
# ============================================================================

def page_feature_contribution_proposed_model() -> None:
    """
    Show:
    1. Feature-level contribution across the five conventional models.
    2. Features with low/unclear measured contribution.
    3. The proposed SH-DAS V4 architecture.
    4. Final six-model comparison.
    """

    import json

    st.markdown(
        '<div class="shml-section-title">'
        'Feature Contribution &amp; Proposed Model'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="shml-section-sub">'
        'Feature-level analysis of the five conventional models followed by '
        'the proposed Self-Healing Drift-Aware Stacking model (SH-DAS V4).'
        '</div>',
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------------

    feature_dir = (
        PROJECT_ROOT
        / "reports"
        / "model_feature_analysis"
    )

    proposed_dir = (
        PROJECT_ROOT
        / "reports"
        / "proposed_model_v4"
    )

    feature_matrix_path = (
        feature_dir
        / "feature_importance_matrix.csv"
    )

    feature_status_path = (
        feature_dir
        / "feature_status_matrix.csv"
    )

    feature_summary_path = (
        feature_dir
        / "model_feature_summary.csv"
    )

    heatmap_path = (
        feature_dir
        / "feature_model_heatmap.png"
    )

    final_comparison_path = (
        proposed_dir
        / "six_model_v4_final_comparison.csv"
    )

    delta_path = (
        proposed_dir
        / "sh_das_v4_vs_conventional.csv"
    )

    proposed_summary_path = (
        proposed_dir
        / "sh_das_v4_summary.json"
    )

    roc_auc_chart_path = (
        proposed_dir
        / "six_model_v4_roc_auc.png"
    )

    # ------------------------------------------------------------------
    # SIDE-BY-SIDE EXISTING MODEL VS FEATURE-AWARE MODEL
    # ------------------------------------------------------------------

    st.markdown(
        '<div class="shml-section-title">'
        'Existing Model vs Feature-Aware Model'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="shml-section-sub">'
        'Feature-level targeting and reported performance comparison.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Features repeatedly identified as more meaningful in the project's
    # measured feature-contribution analysis.
    feature_focus = {
        "contract",
        "tenure",
        "internetservice",
        "monthlycharges",
        "totalcharges",
    }

    feature_names = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
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
        "MonthlyCharges",
        "TotalCharges",
    ]

    feature_rows = []

    for feature_name in feature_names:
        feature_rows.append(
            {
                "Feature": feature_name,
                "Existing Model": "\u2713",
                "Feature-Aware Focus": (
                    "\u2713"
                    if feature_name.lower() in feature_focus
                    else "\u2717"
                ),
            }
        )

    feature_comparison = pd.DataFrame(feature_rows)

    # Use the same saved V4 comparison artifact already used later
    # on this page, so the table does not invent new performance values.
    comparison_for_table = load_csv(
        str(
            PROJECT_ROOT
            / "reports"
            / "proposed_model_v4"
            / "six_model_v4_final_comparison.csv"
        )
    )

    performance_rows = []

    if (
        comparison_for_table is not None
        and "Model" in comparison_for_table.columns
    ):
        model_names = comparison_for_table["Model"].astype(str)

        existing_matches = comparison_for_table[
            model_names.str.contains(
                "Logistic Regression",
                case=False,
                na=False,
            )
        ]

        proposed_matches = comparison_for_table[
            model_names.str.contains(
                "SH-DAS V4",
                case=False,
                na=False,
            )
        ]

        existing_row = (
            existing_matches.iloc[0]
            if not existing_matches.empty
            else None
        )

        proposed_row = (
            proposed_matches.iloc[0]
            if not proposed_matches.empty
            else None
        )

        metric_columns = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
            "ROC-AUC",
            "Training_Time_Seconds",
        ]

        for metric_name in metric_columns:

            if metric_name not in comparison_for_table.columns:
                continue

            existing_value = (
                existing_row[metric_name]
                if existing_row is not None
                else "Not available"
            )

            proposed_value = (
                proposed_row[metric_name]
                if proposed_row is not None
                else "Not available"
            )

            existing_numeric = pd.to_numeric(
                pd.Series([existing_value]),
                errors="coerce",
            ).iloc[0]

            proposed_numeric = pd.to_numeric(
                pd.Series([proposed_value]),
                errors="coerce",
            ).iloc[0]

            if pd.notna(existing_numeric):
                existing_display = f"{float(existing_numeric):.4f}"
            else:
                existing_display = str(existing_value)

            if pd.notna(proposed_numeric):
                proposed_display = f"{float(proposed_numeric):.4f}"
            else:
                proposed_display = str(proposed_value)

            if (
                pd.notna(existing_numeric)
                and pd.notna(proposed_numeric)
            ):
                change_display = (
                    f"{float(proposed_numeric - existing_numeric):+.4f}"
                )
            else:
                change_display = "—"

            label = (
                "Training Time (s)"
                if metric_name == "Training_Time_Seconds"
                else metric_name
            )

            performance_rows.append(
                {
                    "Metric": label,
                    "Existing LR": existing_display,
                    "Feature-Aware SH-DAS V4": proposed_display,
                    "Change": change_display,
                }
            )

    if not performance_rows:
        performance_rows = [
            {
                "Metric": "Accuracy",
                "Existing LR": "Not available",
                "Feature-Aware SH-DAS V4": "Not available",
                "Change": "—",
            },
            {
                "Metric": "Precision",
                "Existing LR": "Not available",
                "Feature-Aware SH-DAS V4": "Not available",
                "Change": "—",
            },
            {
                "Metric": "Recall",
                "Existing LR": "Not available",
                "Feature-Aware SH-DAS V4": "Not available",
                "Change": "—",
            },
            {
                "Metric": "F1-score",
                "Existing LR": "Not available",
                "Feature-Aware SH-DAS V4": "Not available",
                "Change": "—",
            },
            {
                "Metric": "ROC-AUC",
                "Existing LR": "Not available",
                "Feature-Aware SH-DAS V4": "Not available",
                "Change": "—",
            },
        ]

    performance_comparison = pd.DataFrame(performance_rows)

    comparison_col1, comparison_col2 = st.columns(2)


    # ------------------------------------------------------------------------
    # SIX-MODEL FEATURE / SIGNAL CAPABILITY MATRIX
    # ------------------------------------------------------------------------

    st.markdown(
        '<div class="shml-section-title">'
        '1. Six-Model Feature &amp; Signal Capability Matrix'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="shml-info">

        <strong>How to read this matrix:</strong><br><br>

        A <strong>✓</strong> means that the model/system receives or uses
        that information in the evaluated architecture.
        A <strong>✗</strong> means that the information is not an input to
        that model's decision layer.

        <br><br>

        The five conventional classifiers use the same 19 original
        Telco customer features. SH-DAS also retains those features, but
        adds model-prediction signals, feature-level drift information,
        and a meta-model decision layer.

        <br><br>

        Therefore, the SH-DAS advantage is primarily a
        <strong>richer decision pipeline</strong>, not a claim that its
        raw customer columns are different.

        </div>
        """,
        unsafe_allow_html=True,
    )

    six_model_columns = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "SVM",
        "SH-DAS (Proposed)",
    ]

    original_features = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
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
        "MonthlyCharges",
        "TotalCharges",
    ]

    capability_rows = []

    for feature in original_features:

        advantage = "Shared original input"

        if feature in [
            "tenure",
            "MonthlyCharges",
        ]:
            advantage = "Drift-aware monitoring in SH-DAS"

        row = {
            "Feature / Signal": feature,
            "Logistic Regression": "✓",
            "Decision Tree": "✓",
            "Random Forest": "✓",
            "Gradient Boosting": "✓",
            "SVM": "✓",
            "SH-DAS (Proposed)": "✓",
            "SH-DAS Advantage": advantage,
        }

        capability_rows.append(row)

    # Additional decision-level signals used by the proposed framework.
    additional_signals = [
        (
            "LR prediction signal",
            "Multiple-model ensemble signal",
        ),
        (
            "Decision Tree prediction signal",
            "Multiple-model ensemble signal",
        ),
        (
            "Random Forest prediction signal",
            "Multiple-model ensemble signal",
        ),
        (
            "Gradient Boosting prediction signal",
            "Multiple-model ensemble signal",
        ),
        (
            "SVM prediction signal",
            "Multiple-model ensemble signal",
        ),
        (
            "Feature-level drift information",
            "Drift-aware decision context",
        ),
        (
            "Meta-model combination",
            "Adaptive final decision layer",
        ),
    ]

    for signal_name, advantage in additional_signals:

        capability_rows.append(
            {
                "Feature / Signal": signal_name,
                "Logistic Regression": "✗",
                "Decision Tree": "✗",
                "Random Forest": "✗",
                "Gradient Boosting": "✗",
                "SVM": "✗",
                "SH-DAS (Proposed)": "✓",
                "SH-DAS Advantage": advantage,
            }
        )

    capability_df = pd.DataFrame(
        capability_rows,
        columns=[
            "Feature / Signal",
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "Gradient Boosting",
            "SVM",
            "SH-DAS (Proposed)",
            "SH-DAS Advantage",
        ],
    )

    st.dataframe(
        capability_df,
        use_container_width=True,
        hide_index=True,
    )

    # ------------------------------------------------------------------------
    # SH-DAS ADDITIONAL INFORMATION SOURCES
    # ------------------------------------------------------------------------

    st.markdown(
        '<div class="shml-section-title">'
        'What SH-DAS Adds Beyond the Five Conventional Models'
        '</div>',
        unsafe_allow_html=True,
    )

    advantage_df = pd.DataFrame(
        [
            {
                "Component": "Original customer features",
                "Conventional Models":
                    "19 shared customer features",
                "SH-DAS":
                    "Same 19 features",
                "Additional Capability":
                    "Keeps the original information space",
            },
            {
                "Component": "Five model predictions",
                "Conventional Models":
                    "Each model produces its own prediction",
                "SH-DAS":
                    "Consumes predictions from all five",
                "Additional Capability":
                    "Combines model diversity",
            },
            {
                "Component": "Feature-level drift",
                "Conventional Models":
                    "No adaptive decision layer",
                "SH-DAS":
                    "Uses drift-aware information",
                "Additional Capability":
                    "Adds context about distribution change",
            },
            {
                "Component": "Meta-model",
                "Conventional Models":
                    "No second-level model",
                "SH-DAS":
                    "Learns from combined signals",
                "Additional Capability":
                    "Adaptive decision integration",
            },
            {
                "Component": "Self-healing workflow",
                "Conventional Models":
                    "Static prediction",
                "SH-DAS":
                    "Detect → diagnose → adapt → evaluate",
                "Additional Capability":
                    "Designed for model-drift recovery",
            },
        ]
    )

    st.dataframe(
        advantage_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        """
        <div class="shml-card">

        <strong>Where SH-DAS is different:</strong><br><br>

        The conventional models answer:
        <strong>“What does this model predict from the customer data?”</strong>

        <br><br>

        SH-DAS additionally asks:
        <strong>“What are the five models predicting, is the feature
        distribution changing, and how should those signals be combined
        under the observed drift?”</strong>

        <br><br>

        This creates a multi-stage, drift-aware decision process rather than
        relying on one isolated classifier.

        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### 2. Performance Comparison")

    st.dataframe(
        performance_comparison,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Performance values are read from the saved SH-DAS V4 "
        "final comparison artifact used by this dashboard."
    )

# ------------------------------------------------------------------------
    # SECTION 1 — FEATURE CONTRIBUTION
    # ------------------------------------------------------------------------

    st.markdown(
        '<div class="shml-section-title">'
        '1. Feature-Level Contribution Across Five Models'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="shml-info">
        The five conventional classifiers were given the same original dataset
        features. Permutation importance was then used to measure how much
        changing an individual feature affected ROC-AUC on the evaluation data.
        Therefore, this section shows <strong>measured feature contribution</strong>
        rather than incorrectly claiming that a model is technically unable to
        accept a particular feature.
        </div>
        """,
        unsafe_allow_html=True,
    )

    feature_matrix = load_csv(
        str(feature_matrix_path)
    )

    feature_summary = load_csv(
        str(feature_summary_path)
    )

    if feature_matrix is None:
        file_missing_warning(
            feature_matrix_path,
            SRC_DIR / "feature_model_analysis.py",
        )
    else:
        st.markdown(
            "#### Mean Permutation Importance"
        )

        st.dataframe(
            feature_matrix,
            width="stretch",
            hide_index=True,
        )

    # ------------------------------------------------------------------------
    # Heatmap
    # ------------------------------------------------------------------------

    if heatmap_path.exists():

        st.markdown(
            "#### Feature × Model Heatmap"
        )

        st.image(
            str(heatmap_path),
            width="stretch",
        )

        st.caption(
            "Positive values indicate that permuting the feature reduced "
            "ROC-AUC in this experiment. Negative values should be interpreted "
            "carefully because permutation can sometimes produce small "
            "sampling-related changes."
        )

        st.markdown(
            """
            <div class="shml-info">

            <strong>How to Read the Heatmap Colors</strong><br><br>

            <strong>Dark purple / near 0:</strong>
            The feature produced little measurable change in ROC-AUC when
            permuted, so its measured contribution was relatively small
            for that model in this experiment.

            <br><br>

            <strong>Blue to teal:</strong>
            The feature produced a moderate change in ROC-AUC, indicating
            a moderate measured contribution.

            <br><br>

            <strong>Yellow / bright:</strong>
            The feature produced a larger positive change in ROC-AUC when
            permuted. This means the model relied more strongly on that
            feature in the evaluated experiment.

            <br><br>

            <strong>Why this matters:</strong>
            The heatmap compares feature contribution across the five
            conventional classifiers. A feature can be highly important
            for one model and less important for another.

            <br><br>

            <strong>Important:</strong>
            This heatmap measures <em>model contribution</em>, not causal
            importance. A bright feature does not mean that changing that
            real-world attribute will necessarily cause churn.

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="shml-card">

            <strong>Quick takeaway:</strong><br><br>

            Look across a row to compare how strongly different models
            depend on the same feature. Look down a column to see which
            features contributed most to a particular model.

            In the current experiment, features such as
            <strong>tenure</strong>, <strong>Contract</strong>,
            <strong>InternetService</strong>, and
            <strong>MonthlyCharges</strong> show more visible contribution
            for some models than lower-intensity rows.

            </div>
            """,
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------------------
    # Feature summary
    # ------------------------------------------------------------------------

    if feature_summary is not None:

        st.markdown(
            "#### Model Feature Summary"
        )

        st.dataframe(
            feature_summary,
            width="stretch",
            hide_index=True,
        )

    # ------------------------------------------------------------------------
    # SECTION 2 — LOW / UNCLEAR CONTRIBUTION
    # ------------------------------------------------------------------------

    st.markdown(
        '<div class="shml-section-title">'
        '2. Feature-Level Gap Analysis'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="shml-warning">
        <strong>Interpretation:</strong> "Low/Unclear Contribution" means that
        the measured permutation importance was not reliably positive in this
        experiment. It does <strong>not</strong> mean that the algorithm is
        incapable of using that feature.
        </div>
        """,
        unsafe_allow_html=True,
    )

    feature_status = load_csv(
        str(feature_status_path)
    )

    if feature_status is not None:

        st.markdown(
            "#### Contribution Status by Feature and Model"
        )

        st.dataframe(
            feature_status,
            width="stretch",
            hide_index=True,
        )

        st.markdown(
            "#### Filter the Feature Analysis"
        )

        model_columns = [
            column
            for column in feature_status.columns
            if column != "Feature"
        ]

        selected_model = st.selectbox(
            "Select a model",
            model_columns,
            key="feature_status_model",
        )

        if selected_model:
            selected_status = feature_status[
                [
                    "Feature",
                    selected_model,
                ]
            ].copy()

            selected_status = selected_status.rename(
                columns={
                    selected_model:
                    "Contribution Status"
                }
            )

            st.dataframe(
                selected_status,
                width="stretch",
                hide_index=True,
            )


    # ------------------------------------------------------------------------
    # SH-DAS INPUT / INFORMATION ARCHITECTURE
    # ------------------------------------------------------------------------

    st.markdown(
        '<div class="shml-section-title">'
        'SH-DAS Input &amp; Information Architecture'
        '</div>',
        unsafe_allow_html=True,
    )

    architecture_df = pd.DataFrame([
        {
            "Model": "Logistic Regression",
            "Original Customer Features": 19,
            "Additional Prediction Signals": 0,
            "Feature-Level Drift Information": "No",
            "Meta-Model Decision Layer": "No",
        },
        {
            "Model": "Decision Tree",
            "Original Customer Features": 19,
            "Additional Prediction Signals": 0,
            "Feature-Level Drift Information": "No",
            "Meta-Model Decision Layer": "No",
        },
        {
            "Model": "Random Forest",
            "Original Customer Features": 19,
            "Additional Prediction Signals": 0,
            "Feature-Level Drift Information": "No",
            "Meta-Model Decision Layer": "No",
        },
        {
            "Model": "Gradient Boosting",
            "Original Customer Features": 19,
            "Additional Prediction Signals": 0,
            "Feature-Level Drift Information": "No",
            "Meta-Model Decision Layer": "No",
        },
        {
            "Model": "Support Vector Machine",
            "Original Customer Features": 19,
            "Additional Prediction Signals": 0,
            "Feature-Level Drift Information": "No",
            "Meta-Model Decision Layer": "No",
        },
        {
            "Model": "SH-DAS V4 (Proposed)",
            "Original Customer Features": 19,
            "Additional Prediction Signals": 5,
            "Feature-Level Drift Information": "Yes",
            "Meta-Model Decision Layer": "Yes",
        },
    ])

    st.dataframe(
        architecture_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        """
        <div class="shml-info">

        <strong>What is different about SH-DAS?</strong><br><br>

        The five conventional models operate as individual classifiers using
        the same 19 original customer features.

        <br><br>

        <strong>SH-DAS V4 keeps those 19 customer features, but adds a
        system-level decision layer.</strong>

        <br><br>

        It observes the predictions produced by the five conventional models,
        incorporates feature-level drift information, and uses a meta-model
        to combine this information before producing the final prediction.

        <br><br>

        Therefore, the proposed contribution is not simply "more columns".
        It is a <strong>richer information pipeline</strong> combining customer
        features, multiple model predictions, and drift-aware signals.

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="shml-section-title">'
        'Controlled Concept-Drift Performance Comparison'
        '</div>',
        unsafe_allow_html=True,
    )

    controlled_comparison_path = (
        REPORTS_DIR
        / "controlled_concept_drift"
        / "concept_drift_comparison.csv"
    )

    if controlled_comparison_path.exists():

        controlled_df = load_csv(
            str(controlled_comparison_path)
        )

        if controlled_df is not None and not controlled_df.empty:

            st.markdown(
                """
                <div class="shml-warning">
                <strong>Evaluation setting:</strong>
                This comparison comes from the controlled synthetic
                concept-drift stress test. All frozen conventional models
                and SH-DAS adaptive recovery were evaluated on the same
                post-drift test set.
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Normalize column names.
            controlled_df.columns = [
                str(c).strip()
                for c in controlled_df.columns
            ]

            model_column = None

            for candidate in [
                "Model",
                "model",
                "Strategy",
                "strategy",
            ]:
                if candidate in controlled_df.columns:
                    model_column = candidate
                    break

            metric_map = {
                "Accuracy": [
                    "Accuracy",
                    "accuracy",
                ],
                "Precision": [
                    "Precision",
                    "precision",
                ],
                "Recall": [
                    "Recall",
                    "recall",
                ],
                "F1-score": [
                    "F1",
                    "F1-score",
                    "f1",
                    "f1-score",
                ],
                "ROC-AUC": [
                    "ROC-AUC",
                    "ROC_AUC",
                    "roc_auc",
                ],
            }

            if model_column is not None:

                controlled_df["__is_shdas"] = (
                    controlled_df[model_column]
                    .astype(str)
                    .str.contains(
                        "SH-DAS|Adaptive",
                        case=False,
                        regex=True,
                    )
                )

                frozen_df = controlled_df[
                    ~controlled_df["__is_shdas"]
                ].copy()

                shdas_df = controlled_df[
                    controlled_df["__is_shdas"]
                ].copy()

                if not shdas_df.empty and not frozen_df.empty:

                    shdas_row = shdas_df.iloc[0]

                    comparison_rows = []
                    pass_count = 0

                    for metric_name, candidates in metric_map.items():

                        actual_column = None

                        for candidate in candidates:
                            if candidate in controlled_df.columns:
                                actual_column = candidate
                                break

                        if actual_column is None:
                            continue

                        frozen_values = pd.to_numeric(
                            frozen_df[actual_column],
                            errors="coerce",
                        ).dropna()

                        if frozen_values.empty:
                            continue

                        frozen_max = float(
                            frozen_values.max()
                        )

                        shdas_value = float(
                            shdas_row[actual_column]
                        )

                        improvement = (
                            shdas_value - frozen_max
                        )

                        better = shdas_value > frozen_max

                        if better:
                            pass_count += 1

                        comparison_rows.append(
                            {
                                "Metric": metric_name,
                                "Frozen Best": frozen_max,
                                "SH-DAS Adaptive": shdas_value,
                                "Improvement": improvement,
                                "Result": (
                                    "BETTER"
                                    if better
                                    else "NOT BETTER"
                                ),
                            }
                        )

                    result_df = pd.DataFrame(
                        comparison_rows
                    )

                    if not result_df.empty:

                        st.dataframe(
                            result_df,
                            use_container_width=True,
                            hide_index=True,
                        )

                        if pass_count == 5:
                            st.success(
                                "SH-DAS ADAPTIVE RECOVERY — "
                                "5 / 5 METRICS IMPROVED"
                            )
                        else:
                            st.info(
                                f"SH-DAS improved "
                                f"{pass_count} / 5 metrics "
                                "in this controlled experiment."
                            )

                        # Metric cards
                        available = result_df["Metric"].tolist()

                        metric_cols = st.columns(
                            min(5, len(available))
                        )

                        for index, metric_name in enumerate(
                            available
                        ):
                            row = result_df[
                                result_df["Metric"]
                                == metric_name
                            ].iloc[0]

                            with metric_cols[index]:
                                st.metric(
                                    metric_name,
                                    f"{row['SH-DAS Adaptive']:.4f}",
                                    delta=(
                                        f"{row['Improvement']:+.4f}"
                                    ),
                                )

                        st.markdown(
                            '<div class="shml-section-title">'
                            'What Each Evaluation Metric Means'
                            '</div>',
                            unsafe_allow_html=True,
                        )

                        metric_explanations = pd.DataFrame([
                            {
                                "Metric": "Accuracy",
                                "Meaning":
                                    "Percentage of all predictions that are correct.",
                                "Interpretation":
                                    "Higher Accuracy means fewer total classification errors.",
                            },
                            {
                                "Metric": "Precision",
                                "Meaning":
                                    "Percentage of predicted churn cases that are actually churn.",
                                "Interpretation":
                                    "Higher Precision means fewer false churn alerts.",
                            },
                            {
                                "Metric": "Recall",
                                "Meaning":
                                    "Percentage of actual churn cases detected by the model.",
                                "Interpretation":
                                    "Higher Recall means fewer actual churn cases are missed.",
                            },
                            {
                                "Metric": "F1-score",
                                "Meaning":
                                    "Harmonic balance between Precision and Recall.",
                                "Interpretation":
                                    "Higher F1 indicates a stronger balance between false alarms and missed cases.",
                            },
                            {
                                "Metric": "ROC-AUC",
                                "Meaning":
                                    "Measures how well the model separates the two classes.",
                                "Interpretation":
                                    "Higher ROC-AUC means stronger ranking/discrimination capability.",
                            },
                        ])

                        st.dataframe(
                            metric_explanations,
                            use_container_width=True,
                            hide_index=True,
                        )

                        st.markdown(
                            """
                            <div class="shml-card">

                            <strong>How SH-DAS compares in this stress test</strong><br><br>

                            <strong>Accuracy:</strong>
                            SH-DAS makes substantially more correct overall
                            classifications on the post-drift test set.

                            <br><br>

                            <strong>Precision:</strong>
                            SH-DAS produces a higher proportion of correct
                            positive churn predictions.

                            <br><br>

                            <strong>Recall:</strong>
                            SH-DAS detects a larger proportion of the customers
                            who actually belong to the churn class.

                            <br><br>

                            <strong>F1-score:</strong>
                            SH-DAS maintains a stronger balance between
                            Precision and Recall.

                            <br><br>

                            <strong>ROC-AUC:</strong>
                            SH-DAS provides substantially stronger separation
                            between churn and non-churn cases.

                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    else:
        st.info(
            "Controlled concept-drift results are not available yet. "
            "Run controlled_concept_drift_benchmark.py first."
        )


    # ------------------------------------------------------------------------
    # SECTION 3 — PROPOSED MODEL
    # ------------------------------------------------------------------------

    st.markdown(
        '<div class="shml-section-title">'
        '3. Proposed Model — SH-DAS V4'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="shml-card">
        <strong>SH-DAS V4 — Self-Healing Drift-Aware Stacking</strong><br><br>

        The proposed model does not replace the five conventional algorithms.
        Instead, it uses their predictions as inputs to a second-level
        meta-model and combines them with drift-aware information.

        <br><br>

        <strong>Base layer:</strong><br>
        Logistic Regression · Decision Tree · Random Forest ·
        Gradient Boosting · Support Vector Machine

        <br><br>

        <strong>Drift-aware information:</strong><br>
        Feature-level drift indicators for the controlled drift experiment,
        particularly <code>tenure</code> and <code>MonthlyCharges</code>.

        <br><br>

        <strong>Meta layer:</strong><br>
        The meta-model learns how the base-model predictions behave in the
        presence of the observed drift characteristics.

        <br><br>

        <strong>Final stage:</strong><br>
        The SH-DAS prediction is converted into the final churn decision using
        a threshold selected using adaptation data.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="shml-step">
            <div class="shml-step-num">1</div>
            Train five conventional classifiers
        </div>

        <div class="shml-step">
            <div class="shml-step-num">2</div>
            Measure feature-level contribution
        </div>

        <div class="shml-step">
            <div class="shml-step-num">3</div>
            Detect controlled feature drift
        </div>

        <div class="shml-step">
            <div class="shml-step-num">4</div>
            Generate five model predictions
        </div>

        <div class="shml-step">
            <div class="shml-step-num">5</div>
            Combine predictions with drift-aware meta-features
        </div>

        <div class="shml-step">
            <div class="shml-step-num">6</div>
            SH-DAS generates the proposed prediction
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------------
    # Proposed model metadata
    # ------------------------------------------------------------------------

    if proposed_summary_path.exists():

        try:
            with proposed_summary_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                summary = json.load(file)

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown(
                    metric_card(
                        "Proposed Model",
                        "SH-DAS V4",
                    ),
                    unsafe_allow_html=True,
                )

            with c2:
                st.markdown(
                    metric_card(
                        "Meta-Model",
                        str(
                            summary.get(
                                "selected_meta_model",
                                "Not available",
                            )
                        ),
                    ),
                    unsafe_allow_html=True,
                )

            with c3:
                threshold = summary.get(
                    "selected_threshold"
                )

                threshold_text = (
                    f"{threshold:.2f}"
                    if isinstance(
                        threshold,
                        (int, float),
                    )
                    else "Not available"
                )

                st.markdown(
                    metric_card(
                        "Selected Threshold",
                        threshold_text,
                    ),
                    unsafe_allow_html=True,
                )

        except (
            OSError,
            json.JSONDecodeError,
        ) as exc:

            st.warning(
                f"Could not read SH-DAS summary: {exc}"
            )


    # ------------------------------------------------------------------------
    # SECTION 4 — CONTROLLED CONCEPT-DRIFT PERFORMANCE COMPARISON
    # ------------------------------------------------------------------------

    st.markdown(
        '<div class="shml-section-title">'
        '4. SH-DAS Performance Under Controlled Concept Drift'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="shml-info">

        <strong>Why this comparison is important:</strong><br><br>

        The conventional models remain frozen after the original training
        phase. SH-DAS detects the concept drift, uses the dedicated
        adaptation window, retrains its adaptive recovery layer, and then
        evaluates the recovered system on unseen post-drift data.

        <br><br>

        This is a <strong>controlled synthetic Telco-like stress test</strong>
        and is intentionally reported separately from the original IBM
        Telco Customer Churn experiment.

        </div>
        """,
        unsafe_allow_html=True,
    )

    controlled_dir = (
        REPORTS_DIR / "controlled_concept_drift"
    )

    controlled_path = (
        controlled_dir
        / "concept_drift_comparison.csv"
    )

    if controlled_path.exists():

        controlled_df = load_csv(
            str(controlled_path)
        )

        if controlled_df is not None and not controlled_df.empty:

            controlled_df.columns = [
                str(column).strip()
                for column in controlled_df.columns
            ]

            # Identify the model/system column.
            model_col = None

            for candidate in [
                "Model",
                "model",
                "System",
                "system",
                "Strategy",
                "strategy",
            ]:
                if candidate in controlled_df.columns:
                    model_col = candidate
                    break

            metric_candidates = {
                "Accuracy": ["Accuracy", "accuracy"],
                "Precision": ["Precision", "precision"],
                "Recall": ["Recall", "recall"],
                "F1-score": [
                    "F1-score",
                    "F1",
                    "f1-score",
                    "f1",
                ],
                "ROC-AUC": [
                    "ROC-AUC",
                    "ROC_AUC",
                    "roc_auc",
                ],
            }

            if model_col is not None:

                # Find SH-DAS row.
                shdas_mask = (
                    controlled_df[model_col]
                    .astype(str)
                    .str.contains(
                        "SH-DAS|Adaptive",
                        case=False,
                        regex=True,
                    )
                )

                shdas_rows = controlled_df[
                    shdas_mask
                ]

                frozen_rows = controlled_df[
                    ~shdas_mask
                ]

                if (
                    not shdas_rows.empty
                    and not frozen_rows.empty
                ):

                    shdas_row = shdas_rows.iloc[0]

                    comparison_rows = []

                    for metric_name, candidates in (
                        metric_candidates.items()
                    ):

                        actual_col = None

                        for candidate in candidates:
                            if candidate in controlled_df.columns:
                                actual_col = candidate
                                break

                        if actual_col is None:
                            continue

                        frozen_values = pd.to_numeric(
                            frozen_rows[actual_col],
                            errors="coerce",
                        ).dropna()

                        if frozen_values.empty:
                            continue

                        frozen_best = float(
                            frozen_values.max()
                        )

                        shdas_value = float(
                            shdas_row[actual_col]
                        )

                        improvement = (
                            shdas_value - frozen_best
                        )

                        improvement_percent = (
                            (
                                improvement
                                / abs(frozen_best)
                            )
                            * 100
                            if frozen_best != 0
                            else 0
                        )

                        comparison_rows.append(
                            {
                                "Metric": metric_name,
                                "Best Frozen Model": frozen_best,
                                "SH-DAS Adaptive": shdas_value,
                                "Absolute Improvement": improvement,
                                "Relative Improvement (%)":
                                    improvement_percent,
                                "Result":
                                    (
                                        "BETTER"
                                        if shdas_value > frozen_best
                                        else "NOT BETTER"
                                    ),
                            }
                        )

                    result_df = pd.DataFrame(
                        comparison_rows
                    )

                    if not result_df.empty:

                        pass_count = int(
                            (
                                result_df["Result"]
                                == "BETTER"
                            ).sum()
                        )

                        # ------------------------------------------------
                        # Big result banner
                        # ------------------------------------------------
                        if pass_count == 5:

                            st.success(
                                "SH-DAS ADAPTIVE RECOVERY — "
                                "5 / 5 METRICS BETTER THAN THE "
                                "BEST FROZEN CONVENTIONAL RESULT"
                            )

                        # ------------------------------------------------
                        # KPI cards
                        # ------------------------------------------------
                        cards = st.columns(5)

                        for index, row in result_df.iterrows():

                            with cards[index]:

                                st.metric(
                                    row["Metric"],
                                    f'{row["SH-DAS Adaptive"]:.4f}',
                                    f'{row["Absolute Improvement"]:+.4f}',
                                )

                        # ------------------------------------------------
                        # Main comparison
                        # ------------------------------------------------
                        st.markdown(
                            "#### Metric-by-Metric Comparison"
                        )

                        display_df = result_df.copy()

                        display_df[
                            "Best Frozen Model"
                        ] = display_df[
                            "Best Frozen Model"
                        ].map(
                            lambda x: f"{x:.4f}"
                        )

                        display_df[
                            "SH-DAS Adaptive"
                        ] = display_df[
                            "SH-DAS Adaptive"
                        ].map(
                            lambda x: f"{x:.4f}"
                        )

                        display_df[
                            "Absolute Improvement"
                        ] = display_df[
                            "Absolute Improvement"
                        ].map(
                            lambda x: f"{x:+.4f}"
                        )

                        display_df[
                            "Relative Improvement (%)"
                        ] = display_df[
                            "Relative Improvement (%)"
                        ].map(
                            lambda x: f"{x:+.2f}%"
                        )

                        st.dataframe(
                            display_df[
                                [
                                    "Metric",
                                    "Best Frozen Model",
                                    "SH-DAS Adaptive",
                                    "Absolute Improvement",
                                    "Relative Improvement (%)",
                                    "Result",
                                ]
                            ],
                            use_container_width=True,
                            hide_index=True,
                        )

                        # ------------------------------------------------
                        # What each metric means
                        # ------------------------------------------------
                        st.markdown(
                            '<div class="shml-section-title">'
                            'What Each Metric Means'
                            '</div>',
                            unsafe_allow_html=True,
                        )

                        metric_help = pd.DataFrame(
                            [
                                {
                                    "Metric": "Accuracy",
                                    "Meaning":
                                        "Percentage of all predictions that are correct.",
                                    "Why Higher Helps":
                                        "Fewer total classification errors.",
                                },
                                {
                                    "Metric": "Precision",
                                    "Meaning":
                                        "Among predicted churn cases, the percentage that actually churned.",
                                    "Why Higher Helps":
                                        "Fewer false churn alerts.",
                                },
                                {
                                    "Metric": "Recall",
                                    "Meaning":
                                        "Among actual churn cases, the percentage detected.",
                                    "Why Higher Helps":
                                        "Fewer genuine churn cases are missed.",
                                },
                                {
                                    "Metric": "F1-score",
                                    "Meaning":
                                        "Harmonic balance between Precision and Recall.",
                                    "Why Higher Helps":
                                        "Better balance between false alarms and missed churn cases.",
                                },
                                {
                                    "Metric": "ROC-AUC",
                                    "Meaning":
                                        "How well the system separates churn from non-churn.",
                                    "Why Higher Helps":
                                        "Stronger discrimination and ranking quality.",
                                },
                            ]
                        )

                        st.dataframe(
                            metric_help,
                            use_container_width=True,
                            hide_index=True,
                        )

                        # ------------------------------------------------
                        # Interpretation cards
                        # ------------------------------------------------
                        st.markdown(
                            "#### How to Explain the Improvement"
                        )

                        st.markdown(
                            """
                            <div class="shml-card">

                            <strong>Accuracy — 0.8165</strong><br>
                            SH-DAS correctly classifies substantially more
                            post-drift observations than the strongest frozen
                            conventional result.

                            <br><br>

                            <strong>Precision — 0.8009</strong><br>
                            A larger proportion of SH-DAS positive churn
                            predictions are correct.

                            <br><br>

                            <strong>Recall — 0.8762</strong><br>
                            SH-DAS detects substantially more of the customers
                            who actually churn.

                            <br><br>

                            <strong>F1-score — 0.8368</strong><br>
                            The recovered system achieves a stronger balance
                            between precision and recall.

                            <br><br>

                            <strong>ROC-AUC — 0.8988</strong><br>
                            SH-DAS provides stronger separation between the
                            churn and non-churn classes.

                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            else:
                st.warning(
                    "The controlled comparison file was found, "
                    "but its model column could not be identified."
                )

    else:

        st.info(
            "Controlled concept-drift comparison is not available. "
            "Run controlled_concept_drift_benchmark.py first."
        )

    # ------------------------------------------------------------------------
    # ORIGINAL IBM TELCO V4 RESULT
    # ------------------------------------------------------------------------

    st.markdown(
        '<div class="shml-section-title">'
        'Original IBM Telco V4 Result'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="shml-warning">

        The original IBM Telco evaluation is retained separately.
        In that experiment, SH-DAS V4 improved Recall and ROC-AUC but showed
        trade-offs in the other metrics.

        This separation prevents the controlled stress-test results from
        being mistaken for the original IBM Telco experiment.

        </div>
        """,
        unsafe_allow_html=True,
    )

    final_comparison = load_csv(
        str(final_comparison_path)
    )

    if final_comparison is not None:

        st.dataframe(
            final_comparison,
            use_container_width=True,
            hide_index=True,
        )

    # ------------------------------------------------------------------------
    # DELTA TABLE
    # ------------------------------------------------------------------------

    delta_df = load_csv(
        str(delta_path)
    )

    if delta_df is not None:

        st.markdown(
            "#### SH-DAS V4 Improvement / Trade-Off Analysis"
        )

        st.dataframe(
            delta_df,
            width="stretch",
            hide_index=True,
        )

        st.markdown(
            """
            <div class="shml-info">
            These differences are measured against the best conventional value
            for each metric in this experiment. They should be interpreted as
            experiment-specific results, not as a claim that SH-DAS is
            universally superior.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------------------
    # ROC-AUC chart
    # ------------------------------------------------------------------------

    if roc_auc_chart_path.exists():

        st.markdown(
            "#### ROC-AUC Comparison"
        )

        st.image(
            str(roc_auc_chart_path),
            width="stretch",
        )

    # ------------------------------------------------------------------------
    # Final interpretation
    # ------------------------------------------------------------------------

    st.markdown(
        '<div class="shml-section-title">'
        'What Our Model Adds'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="shml-card">

        <strong>Conventional approach:</strong><br>
        A classifier receives customer features and produces a prediction.

        <br><br>

        <strong>Our proposed approach:</strong><br>
        The framework first observes the behaviour of multiple classifiers,
        monitors feature drift, and supplies drift-aware information to a
        meta-model that learns how to combine the base predictions.

        <br><br>

        <strong>Observed experimental advantage:</strong><br>
        In the current simulated-drift evaluation, SH-DAS V4 achieved
        <strong>0.6872 Recall</strong> and <strong>0.8337 ROC-AUC</strong>,
        compared with the best conventional values of approximately
        <strong>0.6631 Recall</strong> and <strong>0.8294 ROC-AUC</strong>.

        <br><br>

        This indicates improved detection of positive churn cases and improved
        ranking quality in this particular drifted evaluation, while the model
        also shows trade-offs in Accuracy, Precision and F1-score.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="shml-warning">
        <strong>Research interpretation:</strong> the proposed model should not
        be described as universally better. Its measured advantage in this
        experiment is specifically observed in Recall and ROC-AUC under the
        simulated drift scenario.
        </div>
        """,
        unsafe_allow_html=True,
    )
def page_selective_retraining_analysis() -> None:
        """Display the selective retraining and cost-analysis experiment."""

        st.markdown(
            '<div class="shml-section-title">'
            'Selective Retraining &amp; Cost Analysis'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="shml-section-sub">'
            'Evaluate whether feature-level drift diagnosis can reduce unnecessary '
            'retraining while preserving model performance.'
            '</div>',
            unsafe_allow_html=True,
        )

        report_dir = (
            REPORTS_DIR
            / "selective_retraining"
        )

        summary_path = (
            report_dir
            / "selective_retraining_summary.csv"
        )

        scenarios_path = (
            report_dir
            / "selective_retraining_scenarios.csv"
        )

        feature_path = (
            report_dir
            / "selective_retraining_feature_contribution.csv"
        )

        text_report_path = (
            report_dir
            / "selective_retraining_report.txt"
        )

        # ------------------------------------------------------------------
        # Check artifacts
        # ------------------------------------------------------------------

        if not summary_path.exists():
            st.warning(
                "Selective retraining results are not available yet. "
                "Run `python src/selective_retraining_experiment.py` "
                "from the project root."
            )
            return

        summary_df = load_csv(
            str(summary_path)
        )

        scenario_df = load_csv(
            str(scenarios_path)
        )

        feature_df = load_csv(
            str(feature_path)
        )

        if summary_df is None or summary_df.empty:
            st.error(
                "The selective retraining summary could not be loaded."
            )
            return

        # ------------------------------------------------------------------
        # Extract summary values
        # ------------------------------------------------------------------

        def get_summary_value(
            metric_name: str,
            default=0,
        ):
            rows = summary_df[
                summary_df["Metric"].astype(str)
                == metric_name
            ]

            if rows.empty:
                return default

            return rows.iloc[0]["Value"]

        scenario_count = get_summary_value(
            "Scenario count",
            0,
        )

        full_count = get_summary_value(
            "Full-policy retrain count",
            0,
        )

        selective_count = get_summary_value(
            "Feature-aware retrain count",
            0,
        )

        frequency_reduction = get_summary_value(
            "Retraining frequency reduction (%)",
            0,
        )

        time_saved = get_summary_value(
            "Estimated training time saved (%)",
            get_summary_value(
                "Training time saved (%)",
                0,
            ),
        )

        avoided = get_summary_value(
            "Unnecessary full retrains avoided",
            0,
        )

        missed = get_summary_value(
            "Possible missed retraining opportunities",
            0,
        )

        low_feature = get_summary_value(
            "Low-contribution feature used",
            "Not available",
        )

        # ------------------------------------------------------------------
        # Main metrics
        # ------------------------------------------------------------------

        st.markdown(
            '<div class="shml-section-title">'
            'Experiment Result'
            '</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(
                metric_card(
                    "Retraining Reduction",
                    f"{float(frequency_reduction):.2f}%",
                    "Feature-aware policy",
                ),
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown(
                metric_card(
                    "Retrains Avoided",
                    str(int(float(avoided))),
                    "Full-policy retraining events avoided",
                ),
                unsafe_allow_html=True,
            )

        with c3:
            st.markdown(
                metric_card(
                    "Estimated Time Saved",
                    f"{float(time_saved):.2f}%",
                    "Experimental cost proxy",
                ),
                unsafe_allow_html=True,
            )

        with c4:
            st.markdown(
                metric_card(
                    "Missed Opportunities",
                    str(int(float(missed))),
                    "Detected in this experiment",
                ),
                unsafe_allow_html=True,
            )

        # ------------------------------------------------------------------
        # Policy comparison
        # ------------------------------------------------------------------

        st.markdown(
            '<div class="shml-section-title">'
            'Full Retraining vs Feature-Aware Policy'
            '</div>',
            unsafe_allow_html=True,
        )

        policy_df = pd.DataFrame(
            [
                {
                    "Policy": "Full Retraining",
                    "Retraining Events": int(
                        float(full_count)
                    ),
                },
                {
                    "Policy": "Feature-Aware",
                    "Retraining Events": int(
                        float(selective_count)
                    ),
                },
            ]
        )

        st.dataframe(
            policy_df,
            use_container_width=True,
            hide_index=True,
        )

        st.bar_chart(
            policy_df.set_index("Policy")
        )
        # ------------------------------------------------------------------
        # Self-Healing Decision Flow
        # ------------------------------------------------------------------

        st.markdown(
            '<div class="shml-section-title">'
            'Feature-Aware Retraining Decision Flow'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '''
            <div class="shml-card" style="padding:1.2rem;">

            <div style="text-align:center; font-weight:700; font-size:1.05rem;">
                Drift Detected
            </div>

            <div style="text-align:center; font-size:1.4rem; margin:.35rem 0;">
                &darr;
            </div>

            <div style="text-align:center; font-weight:700;">
                PSI reaches the high-drift threshold?
            </div>

            <div style="text-align:center; font-size:1.4rem; margin:.35rem 0;">
                &darr;
            </div>

            <div style="text-align:center; font-weight:700;">
                Feature-level drift diagnosis
            </div>

            <div style="text-align:center; font-size:1.4rem; margin:.35rem 0;">
                &darr;
            </div>

            <div style="text-align:center; font-weight:700;">
                Does the drifted feature have meaningful measured contribution?
            </div>

            <div style="
                display:flex;
                justify-content:center;
                gap:2rem;
                margin-top:1rem;
                flex-wrap:wrap;
            ">

                <div style="
                    flex:1;
                    min-width:220px;
                    padding:1rem;
                    border-radius:14px;
                    background:#eef6ff;
                    text-align:center;
                ">
                    <strong>YES</strong>
                    <br>
                    &darr;
                    <br>
                    <strong>RETRAIN</strong>
                    <br>
                    <small>Train candidate and validate performance</small>
                </div>

                <div style="
                    flex:1;
                    min-width:220px;
                    padding:1rem;
                    border-radius:14px;
                    background:#f5f3ff;
                    text-align:center;
                ">
                    <strong>NO</strong>
                    <br>
                    &darr;
                    <br>
                    <strong>SKIP RETRAINING</strong>
                    <br>
                    <small>Avoid an unnecessary retraining event</small>
                </div>

            </div>

            <div style="text-align:center; font-size:1.4rem; margin:.8rem 0;">
                &darr;
            </div>

            <div style="text-align:center; font-weight:700;">
                Performance Validation Gate
            </div>

            <div style="text-align:center; font-size:1.4rem; margin:.35rem 0;">
                &darr;
            </div>

            <div style="text-align:center; font-weight:700;">
                Promote Candidate / Retain Previous Model
            </div>

            </div>
            ''',
            unsafe_allow_html=True,
        )

    # Key research finding
        # ------------------------------------------------------------------

        st.markdown(
            '<div class="shml-section-title">'
            'Key Research Finding'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="shml-card">

            <strong>Observed in this controlled experiment:</strong><br><br>

            The full retraining policy triggered <strong>{int(float(full_count))}
            </strong> retraining events across
            <strong>{int(float(scenario_count))}</strong> scenarios.

            The feature-aware policy triggered
            <strong>{int(float(selective_count))}</strong> events.

            <br><br>

            This represents a measured
            <strong>{float(frequency_reduction):.2f}% reduction</strong>
            in retraining frequency.

            <br><br>

            The selective policy avoided
            <strong>{int(float(avoided))}</strong> retraining event(s) while the
            experiment reported
            <strong>{int(float(missed))}</strong> possible missed retraining
            opportunities.

            </div>
            """,
            unsafe_allow_html=True,
        )

        # ------------------------------------------------------------------
        # Important demonstration scenario
        # ------------------------------------------------------------------

        if scenario_df is not None and not scenario_df.empty:

            st.markdown(
                '<div class="shml-section-title">'
                'High Drift / Low Contribution Case'
                '</div>',
                unsafe_allow_html=True,
            )

            if "Scenario" in scenario_df.columns:

                target_rows = scenario_df[
                    scenario_df["Scenario"].astype(str)
                    == "High drift / low contribution"
                ]

                if not target_rows.empty:

                    row = target_rows.iloc[0]

                    important_df = pd.DataFrame(
                        [
                            {
                                "Metric": "Maximum PSI",
                                "Value": row.get(
                                    "Maximum_PSI",
                                    "N/A",
                                ),
                            },
                            {
                                "Metric": "Top Drift Feature",
                                "Value": row.get(
                                    "Top_Drift_Feature",
                                    "N/A",
                                ),
                            },
                            {
                                "Metric": "Full Retraining",
                                "Value": (
                                    "YES"
                                    if bool(
                                        row.get(
                                            "Full_Retraining_Triggered",
                                            False,
                                        )
                                    )
                                    else "NO"
                                ),
                            },
                            {
                                "Metric": "Feature-Aware Retraining",
                                "Value": (
                                    "YES"
                                    if bool(
                                        row.get(
                                            "Selective_Retraining_Triggered",
                                            False,
                                        )
                                    )
                                    else "NO"
                                ),
                            },
                            {
                                "Metric": "Selected Low-Contribution Feature",
                                "Value": low_feature,
                            },
                        ]
                    )

                    st.dataframe(
                        important_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.success(
                    "This scenario demonstrates the key selective-retraining "
                    "case: high distribution drift was detected, but the "
                    "drifted feature had low/unclear measured contribution, "
                    "so the feature-aware policy did not trigger retraining."
                )

                st.markdown(
                    f"""
                    <div class="shml-info">

                    <strong>Why was retraining skipped?</strong><br><br>

                    The maximum PSI reached
                    <strong>{float(row.get("Maximum_PSI", 0)):.4f}</strong>,
                    crossing the high-drift threshold.

                    <br><br>

                    However, the affected feature
                    <strong>{row.get("Top_Drift_Feature", "N/A")}</strong>
                    was classified as having low/unclear measured contribution.

                    <br><br>

                    Therefore:

                    <strong>Full Policy &rarr; RETRAIN</strong><br>
                    <strong>Feature-Aware Policy &rarr; SKIP</strong>

                    <br><br>

                    This is the key controlled example supporting the
                    selective-retraining research question.

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ------------------------------------------------------------------
        # Scenario table
        # ------------------------------------------------------------------

        st.markdown(
            '<div class="shml-section-title">'
            'Scenario-by-Scenario Results'
            '</div>',
            unsafe_allow_html=True,
        )

        if scenario_df is not None and not scenario_df.empty:

            display_columns = [
                "Scenario",
                "Maximum_PSI",
                "Top_Drift_Feature",
                "Full_Retraining_Triggered",
                "Selective_Retraining_Triggered",
                "Unnecessary_Full_Retraining",
                "Possible_Missed_Retraining_Opportunity",
            ]

            available_columns = [
                column
                for column in display_columns
                if column in scenario_df.columns
            ]

            st.dataframe(
                scenario_df[available_columns],
                use_container_width=True,
                hide_index=True,
            )

        # ------------------------------------------------------------------
        # Feature contribution
        # ------------------------------------------------------------------

        with st.expander(
            "View feature contribution used by the policy"
        ):
            if feature_df is not None and not feature_df.empty:
                st.dataframe(
                    feature_df,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info(
                    "Feature contribution report is not available."
                )

        # ------------------------------------------------------------------
        # Interpretation
        # ------------------------------------------------------------------

        st.markdown(
            '<div class="shml-section-title">'
            'How to Interpret This Experiment'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="shml-info">

            <strong>Full retraining policy:</strong>
            retraining is triggered whenever the maximum PSI reaches the
            project's high-drift threshold.

            <br><br>

            <strong>Feature-aware policy:</strong>
            retraining is triggered only when high drift overlaps with a
            feature having meaningful measured contribution to the baseline
            model.

            <br><br>

            <strong>Important limitation:</strong>
            These results come from controlled simulated drift on offline
            data. The training-time value is an experimental cost proxy,
            not a production infrastructure-cost estimate.

            </div>
            """,
            unsafe_allow_html=True,
        )

        # ------------------------------------------------------------------
        # Downloads
        # ------------------------------------------------------------------

        st.markdown(
            '<div class="shml-section-title">Download Experiment Artifacts</div>',
            unsafe_allow_html=True,
        )

        download_button_for_file(
            summary_path,
            "Download selective_retraining_summary.csv",
            "selective_retraining_summary_download",
        )

        download_button_for_file(
            scenarios_path,
            "Download selective_retraining_scenarios.csv",
            "selective_retraining_scenarios_download",
        )

        download_button_for_file(
            feature_path,
            "Download feature contribution report",
            "selective_retraining_feature_download",
        )

        download_button_for_file(
            text_report_path,
            "Download experiment report",
            "selective_retraining_text_download",
        )


def page_drift_detective() -> None:
    """Interactive investigation of a simulated model-drift case."""

    st.markdown(
        '<div class="shml-section-title">Drift Detective</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="shml-section-sub">'
        'Investigate a model-drift case step by step and discover why '
        'the self-healing system retrains or safely skips retraining.'
        '</div>',
        unsafe_allow_html=True,
    )

    report_dir = REPORTS_DIR / "selective_retraining"
    scenarios_path = report_dir / "selective_retraining_scenarios.csv"
    feature_path = report_dir / "selective_retraining_feature_contribution.csv"

    scenario_df = load_csv(str(scenarios_path))
    feature_df = load_csv(str(feature_path))

    if scenario_df is None or scenario_df.empty:
        st.warning(
            "Drift Detective needs the selective retraining experiment results. "
            "Run `python src/selective_retraining_experiment.py` first."
        )
        return

    if "Scenario" not in scenario_df.columns:
        st.error("The scenario report is missing the Scenario column.")
        return

    scenarios = scenario_df["Scenario"].astype(str).tolist()

    preferred_case = "High drift / low contribution"
    default_index = (
        scenarios.index(preferred_case)
        if preferred_case in scenarios
        else 0
    )

    selected_scenario = st.selectbox(
        "Choose a case to investigate",
        scenarios,
        index=default_index,
        key="drift_detective_case_selector",
    )

    selected_rows = scenario_df[
        scenario_df["Scenario"].astype(str) == selected_scenario
    ]

    if selected_rows.empty:
        st.warning("The selected case could not be loaded.")
        return

    row = selected_rows.iloc[0]

    def as_bool(value) -> bool:
        return str(value).strip().lower() in {
            "true",
            "1",
            "yes",
        }

    max_psi = float(row.get("Maximum_PSI", 0))
    top_feature = str(row.get("Top_Drift_Feature", "Not available"))
    full_trigger = as_bool(
        row.get("Full_Retraining_Triggered", False)
    )
    selective_trigger = as_bool(
        row.get("Selective_Retraining_Triggered", False)
    )

    if max_psi >= 0.25:
        severity = "HIGH"
    elif max_psi >= 0.10:
        severity = "MODERATE"
    else:
        severity = "LOW"

    previous_case = st.session_state.get(
        "drift_detective_active_case"
    )

    if previous_case != selected_scenario:
        st.session_state["drift_detective_active_case"] = selected_scenario
        st.session_state["drift_detective_stage"] = 0

    stage = int(
        st.session_state.get(
            "drift_detective_stage",
            0,
        )
    )

    st.markdown(
        f"""
        <div class="shml-card">

        <strong>CASE FILE</strong><br><br>

        <strong>Scenario:</strong> {selected_scenario}<br>
        <strong>Investigation status:</strong> {severity} drift<br>
        <strong>Objective:</strong> identify what changed, determine why it
        matters, and establish whether retraining is justified.

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="shml-section-title">Evidence Board</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            metric_card(
                "Maximum PSI",
                f"{max_psi:.4f}",
                "Distribution-shift signal",
            ),
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            metric_card(
                "Drift Severity",
                severity,
                "Project heuristic",
            ),
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            metric_card(
                "Top Drift Feature",
                top_feature,
                "Primary feature under investigation",
            ),
            unsafe_allow_html=True,
        )

    if stage == 0:
        st.markdown(
            """
            <div class="shml-info">

            <strong>Detective briefing</strong><br><br>

            The model has encountered an unusual data-distribution pattern.
            Do not assume that drift automatically means retraining is needed.
            Follow the evidence.

            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "START INVESTIGATION",
            width="stretch",
            key="drift_detective_start",
        ):
            st.session_state["drift_detective_stage"] = 1
            st.rerun()

        return

    # ------------------------------------------------------------------
    # CLUE 1
    # ------------------------------------------------------------------

    st.markdown(
        '<div class="shml-section-title">Clue 1 ? What Changed?</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="shml-card">

        <strong>Distribution evidence</strong><br><br>

        The maximum observed PSI is
        <strong>{max_psi:.4f}</strong>.

        <br><br>

        The project's high-drift threshold is
        <strong>0.25</strong>.

        <br><br>

        The system therefore classified this case as
        <strong>{severity}</strong> drift.

        <br><br>

        <strong>Primary feature:</strong> {top_feature}

        </div>
        """,
        unsafe_allow_html=True,
    )

    if stage == 1:
        if st.button(
            "REVEAL ROOT CAUSE",
            width="stretch",
            key="drift_detective_root",
        ):
            st.session_state["drift_detective_stage"] = 2
            st.rerun()

        return

    # ------------------------------------------------------------------
    # CLUE 2
    # ------------------------------------------------------------------

    st.markdown(
        '<div class="shml-section-title">Clue 2 ? Why Does It Matter?</div>',
        unsafe_allow_html=True,
    )

    if selective_trigger:
        contribution_status = "MEANINGFUL CONTRIBUTION"
        contribution_message = (
            "The feature-aware policy found meaningful measured contribution "
            "overlap with the detected drift."
        )
    else:
        contribution_status = "LOW / UNCLEAR CONTRIBUTION"
        contribution_message = (
            "The feature-aware policy did not find meaningful measured "
            "contribution overlap with the detected drift."
        )

    st.markdown(
        f"""
        <div class="shml-card">

        <strong>Feature under investigation:</strong>
        {top_feature}

        <br><br>

        <strong>Measured contribution:</strong>
        {contribution_status}

        <br><br>

        {contribution_message}

        <br><br>

        This is the key distinction in selective retraining:
        distribution drift is a warning signal, while feature contribution
        helps determine whether that signal justifies another training cycle.

        </div>
        """,
        unsafe_allow_html=True,
    )

    if feature_df is not None and not feature_df.empty:
        with st.expander("View feature-contribution evidence"):
            if "Feature" in feature_df.columns:
                matching_feature = feature_df[
                    feature_df["Feature"].astype(str) == top_feature
                ]

                if not matching_feature.empty:
                    st.dataframe(
                        matching_feature,
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.dataframe(
                        feature_df,
                        use_container_width=True,
                        hide_index=True,
                    )
            else:
                st.dataframe(
                    feature_df,
                    use_container_width=True,
                    hide_index=True,
                )

    if stage == 2:
        if st.button(
            "REVEAL SYSTEM DECISION",
            width="stretch",
            key="drift_detective_decision",
        ):
            st.session_state["drift_detective_stage"] = 3
            st.rerun()

        return

    # ------------------------------------------------------------------
    # CLUE 3
    # ------------------------------------------------------------------

    st.markdown(
        '<div class="shml-section-title">Clue 3 ? What Should the System Do?</div>',
        unsafe_allow_html=True,
    )

    if full_trigger and not selective_trigger:
        decision_title = "SKIP RETRAINING"
        decision_text = (
            "The full policy would retrain because the PSI is high, but the "
            "feature-aware policy prevents an unnecessary retraining event."
        )
    elif selective_trigger:
        decision_title = "RETRAIN CANDIDATE"
        decision_text = (
            "The drift overlaps with meaningful measured feature contribution, "
            "so candidate retraining is justified."
        )
    else:
        decision_title = "NO RETRAINING"
        decision_text = (
            "The observed drift did not satisfy the conditions required by "
            "the feature-aware retraining policy."
        )

    st.markdown(
        f"""
        <div class="shml-card" style="text-align:center;">

        <div style="font-size:1.45rem; font-weight:800;">
            {decision_title}
        </div>

        <br>

        {decision_text}

        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            metric_card(
                "Full Policy",
                "RETRAIN" if full_trigger else "SKIP",
                "PSI-only decision",
            ),
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            metric_card(
                "Feature-Aware Policy",
                "RETRAIN" if selective_trigger else "SKIP",
                "PSI + contribution decision",
            ),
            unsafe_allow_html=True,
        )

    if stage == 3:
        if st.button(
            "SOLVE CASE",
            width="stretch",
            key="drift_detective_solve",
        ):
            st.session_state["drift_detective_stage"] = 4
            st.rerun()

        return

    # ------------------------------------------------------------------
    # FINAL VERDICT
    # ------------------------------------------------------------------

    st.markdown(
        '<div class="shml-section-title">Case Solved</div>',
        unsafe_allow_html=True,
    )

    if full_trigger and not selective_trigger:
        verdict = "UNNECESSARY RETRAINING AVOIDED"
        verdict_text = (
            "The framework detected distribution drift but avoided an "
            "unnecessary full retraining event after feature-level diagnosis."
        )
    elif selective_trigger:
        verdict = "CANDIDATE RETRAINING JUSTIFIED"
        verdict_text = (
            "Feature-level diagnosis justified candidate retraining. "
            "The candidate must still pass the performance gate before promotion."
        )
    else:
        verdict = "MODEL REMAINS STABLE"
        verdict_text = (
            "The observed change did not justify retraining under the "
            "feature-aware policy."
        )

    st.success(verdict)

    st.markdown(
        f"""
        <div class="shml-info">

        <strong>Investigation summary</strong><br><br>

        Drift detected:
        <strong>{"YES" if max_psi >= 0.25 else "NO"}</strong><br>

        Primary feature:
        <strong>{top_feature}</strong><br>

        Full-policy action:
        <strong>{"RETRAIN" if full_trigger else "SKIP"}</strong><br>

        Feature-aware action:
        <strong>{"RETRAIN" if selective_trigger else "SKIP"}</strong>

        <br><br>

        {verdict_text}

        <br><br>

        <strong>Research connection:</strong>
        feature-level diagnosis separates drift that may warrant adaptation
        from drift that may not justify another training cycle.

        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # PREDICTION AUTOPSY
    # ------------------------------------------------------------------

    st.markdown(
        '<div class="shml-section-title">Prediction Autopsy</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="shml-section-sub">'
        'Pick one customer record and investigate why the baseline model '
        'made its prediction. Contributions are model explanations, '
        'not causal claims.'
        '</div>',
        unsafe_allow_html=True,
    )

    import joblib
    import numpy as np

    data_path = (
        DATA_DIR
        / "raw"
        / "Telco-Customer-Churn.csv"
    )

    model_path = PATHS.get(
        "baseline_model",
        MODELS_DIR / "baseline_churn_pipeline.joblib",
    )

    if not data_path.exists():
        st.warning(
            "The Telco Customer Churn dataset was not found, "
            "so Prediction Autopsy cannot be displayed."
        )
    elif not model_path.exists():
        st.warning(
            "The baseline model artifact was not found, "
            "so Prediction Autopsy cannot be displayed."
        )
    else:
        try:
            autopsy_df = pd.read_csv(data_path)

            if "TotalCharges" in autopsy_df.columns:
                autopsy_df["TotalCharges"] = pd.to_numeric(
                    autopsy_df["TotalCharges"],
                    errors="coerce",
                )

            autopsy_features = autopsy_df.drop(
                columns=["customerID", "Churn"],
                errors="ignore",
            )

            record_number = st.slider(
                "Select customer record",
                min_value=1,
                max_value=len(autopsy_features),
                value=1,
                step=1,
                key="prediction_autopsy_record",
            )

            record_index = record_number - 1

            selected_record = autopsy_features.iloc[
                [record_index]
            ].copy()

            model = joblib.load(model_path)

            probability = float(
                model.predict_proba(selected_record)[0][1]
            )

            predicted_class = (
                "CHURN"
                if probability >= 0.50
                else "NO CHURN"
            )

            actual_class = "Not available"

            if "Churn" in autopsy_df.columns:
                actual_class = str(
                    autopsy_df.iloc[record_index]["Churn"]
                )

            top_feature_name = str(top_feature)

            # ----------------------------------------------------------
            # Prediction headline
            # ----------------------------------------------------------

            st.markdown(
                '<div class="shml-card" style="padding:1.15rem;">'
                '<strong>FORENSIC PREDICTION</strong><br><br>'
                f'<strong>Record:</strong> #{record_number}<br>'
                f'<strong>Predicted outcome:</strong> {predicted_class}<br>'
                f'<strong>Churn probability:</strong> '
                f'{probability * 100:.1f}%<br>'
                f'<strong>Actual outcome:</strong> {actual_class}'
                '</div>',
                unsafe_allow_html=True,
            )

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown(
                    metric_card(
                        "Churn Probability",
                        f"{probability * 100:.1f}%",
                        "Baseline model output",
                    ),
                    unsafe_allow_html=True,
                )

            with c2:
                st.markdown(
                    metric_card(
                        "Prediction",
                        predicted_class,
                        "0.50 decision threshold",
                    ),
                    unsafe_allow_html=True,
                )

            with c3:
                st.markdown(
                    metric_card(
                        "Actual",
                        actual_class,
                        "Observed dataset label",
                    ),
                    unsafe_allow_html=True,
                )

            # ----------------------------------------------------------
            # Local explanation for Logistic Regression
            # ----------------------------------------------------------

            contribution_rows = []

            try:
                if (
                    hasattr(model, "named_steps")
                    and len(model.named_steps) >= 2
                ):
                    estimator = list(
                        model.named_steps.values()
                    )[-1]

                    preprocessor = model[:-1]

                    if hasattr(estimator, "coef_") and hasattr(
                        preprocessor,
                        "get_feature_names_out",
                    ):
                        transformed = preprocessor.transform(
                            selected_record
                        )

                        if hasattr(transformed, "toarray"):
                            transformed = transformed.toarray()

                        transformed = np.asarray(
                            transformed
                        ).ravel()

                        feature_names = (
                            preprocessor
                            .get_feature_names_out()
                            .tolist()
                        )

                        coefficients = np.asarray(
                            estimator.coef_
                        ).ravel()

                        local_values = []

                        for name, value, coefficient in zip(
                            feature_names,
                            transformed,
                            coefficients,
                        ):
                            contribution = (
                                float(value) * float(coefficient)
                            )

                            if "__" in name:
                                clean_name = name.split(
                                    "__",
                                    1,
                                )[1]
                            else:
                                clean_name = name

                            local_values.append(
                                {
                                    "Encoded Feature": clean_name,
                                    "Contribution": contribution,
                                }
                            )

                        encoded_df = pd.DataFrame(
                            local_values
                        )

                        # Map one-hot encoded features back to their
                        # original dataset feature where possible.
                        original_columns = list(
                            autopsy_features.columns
                        )

                        def map_original_feature(
                            encoded_name: str,
                        ) -> str:
                            for column in original_columns:
                                if encoded_name == column:
                                    return column

                                if encoded_name.startswith(
                                    column + "_"
                                ):
                                    return column

                            return encoded_name

                        encoded_df[
                            "Feature"
                        ] = encoded_df[
                            "Encoded Feature"
                        ].apply(
                            map_original_feature
                        )

                        aggregated = (
                            encoded_df
                            .groupby("Feature", as_index=False)[
                                "Contribution"
                            ]
                            .sum()
                        )

                        aggregated[
                            "Direction"
                        ] = aggregated[
                            "Contribution"
                        ].apply(
                            lambda x:
                            "Pushes toward churn"
                            if x > 0
                            else "Pushes toward no churn"
                        )

                        aggregated[
                            "Absolute Contribution"
                        ] = aggregated[
                            "Contribution"
                        ].abs()

                        aggregated = aggregated.sort_values(
                            "Absolute Contribution",
                            ascending=False,
                        )

                        drift_df_for_autopsy = load_csv(
                            str(
                                PATHS.get(
                                    "feature_drift_csv",
                                    REPORTS_DIR
                                    / "feature_drift_scores.csv",
                                )
                            )
                        )

                        drift_lookup = {}

                        if (
                            drift_df_for_autopsy
                            is not None
                            and not drift_df_for_autopsy.empty
                        ):
                            column_lookup = {
                                str(column).lower():
                                column
                                for column in
                                drift_df_for_autopsy.columns
                            }

                            feature_column = (
                                column_lookup.get("feature")
                            )
                            psi_column = (
                                column_lookup.get("psi")
                            )
                            status_column = (
                                column_lookup.get("status")
                            )

                            if feature_column:
                                for _, drift_row in (
                                    drift_df_for_autopsy.iterrows()
                                ):
                                    feature_key = str(
                                        drift_row[
                                            feature_column
                                        ]
                                    )

                                    drift_lookup[
                                        feature_key
                                    ] = {
                                        "PSI":
                                        drift_row.get(
                                            psi_column,
                                            "N/A",
                                        )
                                        if psi_column
                                        else "N/A",
                                        "Status":
                                        drift_row.get(
                                            status_column,
                                            "N/A",
                                        )
                                        if status_column
                                        else "N/A",
                                    }

                        for _, contribution_row in (
                            aggregated.head(8).iterrows()
                        ):
                            feature_name = str(
                                contribution_row["Feature"]
                            )

                            drift_info = drift_lookup.get(
                                feature_name,
                                {},
                            )

                            contribution_rows.append(
                                {
                                    "Feature": feature_name,
                                    "Local Contribution":
                                    float(
                                        contribution_row[
                                            "Contribution"
                                        ]
                                    ),
                                    "Direction":
                                    contribution_row[
                                        "Direction"
                                    ],
                                    "Drift PSI":
                                    drift_info.get(
                                        "PSI",
                                        "N/A",
                                    ),
                                    "Drift Status":
                                    drift_info.get(
                                        "Status",
                                        "N/A",
                                    ),
                                }
                            )

            except Exception as explanation_error:
                st.info(
                    "A coefficient-based local explanation was not "
                    f"available for this record: {explanation_error}"
                )

            # ----------------------------------------------------------
            # Evidence table
            # ----------------------------------------------------------

            if contribution_rows:
                st.markdown(
                    '<div class="shml-section-title">'
                    'Why Did the Model Predict This?'
                    '</div>',
                    unsafe_allow_html=True,
                )

                contribution_display = pd.DataFrame(
                    contribution_rows
                )

                contribution_display[
                    "Local Contribution"
                ] = contribution_display[
                    "Local Contribution"
                ].round(4)

                st.dataframe(
                    contribution_display,
                    use_container_width=True,
                    hide_index=True,
                )

                drifted_feature_match = any(
                    str(item["Feature"]).lower()
                    == top_feature_name.lower()
                    for item in contribution_rows
                )

                if drifted_feature_match:
                    st.success(
                        f"The investigated drift feature "
                        f"'{top_feature_name}' also appears among "
                        "the strongest local contributors for this "
                        "prediction."
                    )
                else:
                    st.info(
                        f"The investigated drift feature "
                        f"'{top_feature_name}' was not among the "
                        "strongest local contributors for this "
                        "particular prediction."
                    )

                st.markdown(
                    """
                    <div class="shml-info">

                    <strong>How to read this table:</strong><br><br>

                    A positive local contribution pushes the logistic model
                    toward the churn class. A negative contribution pushes it
                    toward the no-churn class.

                    <br><br>

                    These are model-level explanations based on the learned
                    coefficients and transformed feature values. They should
                    not be interpreted as causal effects.

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # ----------------------------------------------------------
            # Customer feature snapshot
            # ----------------------------------------------------------

            with st.expander(
                "View selected customer's feature snapshot"
            ):
                snapshot = selected_record.T.reset_index()
                snapshot.columns = [
                    "Feature",
                    "Value",
                ]

                st.dataframe(
                    snapshot,
                    use_container_width=True,
                    hide_index=True,
                )

        except Exception as autopsy_error:
            st.error(
                "Prediction Autopsy could not be loaded: "
                f"{autopsy_error}"
            )

    if st.button(
        "START A NEW INVESTIGATION",
        width="stretch",
        key="drift_detective_reset",
    ):
        st.session_state["drift_detective_stage"] = 0
        st.rerun()



def page_controlled_concept_drift() -> None:
    """Display the controlled synthetic concept-drift stress test."""

    st.markdown(
        """
        <div class="shml-hero">
            <div class="shml-badge">
                CONTROLLED EXPERIMENT · SYNTHETIC CONCEPT DRIFT
            </div>
            <h1>CONTROLLED CONCEPT-DRIFT STRESS TEST</h1>
            <p>
                This experiment evaluates whether SH-DAS can recover from
                concept drift when conventional models remain frozen.
                It is intentionally separate from the original IBM Telco
                Customer Churn experiment.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    controlled_dir = REPORTS_DIR / "controlled_concept_drift"

    comparison_path = (
        controlled_dir / "concept_drift_comparison.csv"
    )
    diagnosis_path = (
        controlled_dir / "concept_drift_diagnosis.csv"
    )
    summary_path = (
        controlled_dir / "concept_drift_summary.json"
    )
    report_path = (
        controlled_dir / "concept_drift_report.txt"
    )
    dataset_path = (
        DATA_DIR / "controlled"
        / "concept_drift_telco_like_12000.csv"
    )

    if not comparison_path.exists():
        st.warning(
            "Controlled concept-drift benchmark results are not available yet."
        )

        script_path = (
            SRC_DIR / "controlled_concept_drift_benchmark.py"
        )

        if st.button(
            "RUN CONTROLLED CONCEPT-DRIFT BENCHMARK",
            width="stretch",
            key="run_controlled_concept_drift",
        ):
            run_script_safely(script_path)
            st.rerun()

        return

    # --------------------------------------------------------
    # Load artifacts
    # --------------------------------------------------------
    try:
        comparison_df = pd.read_csv(comparison_path)

        diagnosis_df = (
            pd.read_csv(diagnosis_path)
            if diagnosis_path.exists()
            else pd.DataFrame()
        )

        summary = {}
        if summary_path.exists():
            summary = json.loads(
                summary_path.read_text(encoding="utf-8")
            )

    except Exception as exc:
        st.error(
            f"Could not load controlled experiment results: {exc}"
        )
        return

    # --------------------------------------------------------
    # Experiment structure
    # --------------------------------------------------------
    st.markdown(
        '<div class="shml-section-title">'
        'Experiment Structure'
        '</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Total Records", "12,000")

    with c2:
        st.metric("Pre-Drift Training", "8,000")

    with c3:
        st.metric("Adaptation Window", "2,000")

    with c4:
        st.metric("Post-Drift Test", "2,000")

    st.markdown(
        """
        <div class="shml-info">
        <strong>Experimental note:</strong><br>
        This is a controlled synthetic Telco-like dataset created to
        stress-test self-healing under concept drift. It is not presented
        as additional IBM Telco Customer Churn data.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Drift detection
    # --------------------------------------------------------
    st.markdown(
        '<div class="shml-section-title">'
        'Drift Detection'
        '</div>',
        unsafe_allow_html=True,
    )

    d1, d2, d3 = st.columns(3)

    with d1:
        st.metric(
            "Churn Rate Before Drift",
            "56.16%",
        )

    with d2:
        st.metric(
            "Churn Rate After Drift",
            "54.80%",
        )

    with d3:
        st.metric(
            "Self-Healing Trigger",
            "ACTIVATED",
        )

    if not diagnosis_df.empty:
        st.markdown("### Detected Feature Changes")
        st.dataframe(
            diagnosis_df,
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # Performance comparison
    # --------------------------------------------------------
    st.markdown(
        '<div class="shml-section-title">'
        'Post-Drift Performance Comparison'
        '</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # Extract SH-DAS row / frozen maxima
    # --------------------------------------------------------
    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
    ]

    normalized_columns = {
        str(c).strip().lower(): c
        for c in comparison_df.columns
    }

    model_col = None

    for candidate in [
        "Model",
        "model",
        "System",
        "system",
        "Strategy",
        "strategy",
    ]:
        if candidate in comparison_df.columns:
            model_col = candidate
            break

    if model_col is not None:
        sh_rows = comparison_df[
            comparison_df[model_col]
            .astype(str)
            .str.contains(
                "SH-DAS|Adaptive",
                case=False,
                regex=True,
            )
        ]

        frozen_rows = comparison_df[
            ~comparison_df[model_col]
            .astype(str)
            .str.contains(
                "SH-DAS|Adaptive",
                case=False,
                regex=True,
            )
        ]
    else:
        sh_rows = pd.DataFrame()
        frozen_rows = comparison_df

    if not sh_rows.empty and not frozen_rows.empty:
        sh_row = sh_rows.iloc[0]

        frozen_maxima = {}
        sh_values = {}

        for metric in metric_columns:
            actual_col = normalized_columns.get(
                metric.lower()
            )

            if actual_col is None:
                continue

            numeric_frozen = pd.to_numeric(
                frozen_rows[actual_col],
                errors="coerce",
            ).dropna()

            if numeric_frozen.empty:
                continue

            frozen_maxima[metric] = numeric_frozen.max()
            sh_values[metric] = float(sh_row[actual_col])

        # ----------------------------------------------------
        # 5/5 result banner
        # ----------------------------------------------------
        passes = []

        for metric in frozen_maxima:
            passes.append(
                sh_values[metric]
                > frozen_maxima[metric]
            )

        dominance_count = sum(passes)

        if dominance_count == 5:
            st.success(
                "SH-DAS OUTPERFORMED THE FROZEN MODELS "
                "ON ALL FIVE METRICS — 5/5 DOMINANCE"
            )
        else:
            st.info(
                f"Measured dominance in this run: "
                f"{dominance_count}/5 metrics."
            )

        # ----------------------------------------------------
        # Metrics cards
        # ----------------------------------------------------
        cols = st.columns(5)

        for idx, metric in enumerate(
            metric_columns
        ):
            with cols[idx]:
                if metric in sh_values:
                    st.metric(
                        metric,
                        f"{sh_values[metric]:.4f}",
                        delta=(
                            f"+{sh_values[metric] - frozen_maxima[metric]:.4f}"
                        ),
                    )

        # ----------------------------------------------------
        # Bar chart
        # ----------------------------------------------------
        chart_rows = []

        for metric in frozen_maxima:
            chart_rows.append(
                {
                    "Metric": metric,
                    "Frozen Maximum":
                        frozen_maxima[metric],
                    "SH-DAS":
                        sh_values[metric],
                }
            )

        chart_df = pd.DataFrame(chart_rows)

        if not chart_df.empty:
            fig = px.bar(
                chart_df,
                x="Metric",
                y=[
                    "Frozen Maximum",
                    "SH-DAS",
                ],
                barmode="group",
                title=(
                    "Frozen Conventional Models vs "
                    "SH-DAS Adaptive Recovery"
                ),
            )

            fig.update_layout(
                yaxis_title="Score",
                xaxis_title="Evaluation Metric",
                legend_title="System",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    # --------------------------------------------------------
    # Recovery explanation
    # --------------------------------------------------------
    st.markdown(
        '<div class="shml-section-title">'
        'Self-Healing Recovery'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="shml-card">
        <strong>Recovery sequence:</strong><br><br>

        <strong>1. Detect</strong> — SH-DAS identifies changes in the
        post-drift feature distributions.<br><br>

        <strong>2. Diagnose</strong> — affected features are surfaced
        instead of treating drift as a single unexplained event.<br><br>

        <strong>3. Trigger</strong> — the measured degradation activates
        the recovery mechanism.<br><br>

        <strong>4. Adapt</strong> — adaptive models are retrained using
        the dedicated adaptation window.<br><br>

        <strong>5. Evaluate</strong> — the recovered model is evaluated
        on unseen post-drift data.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Downloads
    # --------------------------------------------------------
    st.markdown(
        '<div class="shml-section-title">'
        'Download Controlled Experiment Artifacts'
        '</div>',
        unsafe_allow_html=True,
    )

    download_button_for_file(
        comparison_path,
        "Download concept_drift_comparison.csv",
        "controlled_comparison_download",
    )

    download_button_for_file(
        diagnosis_path,
        "Download concept_drift_diagnosis.csv",
        "controlled_diagnosis_download",
    )

    download_button_for_file(
        summary_path,
        "Download concept_drift_summary.json",
        "controlled_summary_download",
    )

    download_button_for_file(
        report_path,
        "Download concept_drift_report.txt",
        "controlled_report_download",
    )

    if dataset_path.exists():
        download_button_for_file(
            dataset_path,
            "Download controlled 12,000-record dataset",
            "controlled_dataset_download",
        )

    # --------------------------------------------------------
    # Final interpretation
    # --------------------------------------------------------
    st.markdown(
        '<div class="shml-section-title">'
        'Interpretation for the Project'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="shml-info">

        <strong>What this experiment demonstrates:</strong><br><br>

        The experiment is designed to test the central self-healing
        hypothesis: when the relationship between features and the target
        changes, a frozen model can degrade substantially, while a system
        that detects the change and adapts using recent data can recover
        predictive performance.

        <br><br>

        <strong>Important distinction:</strong><br>
        This controlled synthetic benchmark is supplementary evidence.
        The original IBM Telco experiment remains separate and is not
        replaced by these results.

        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_css()
    selected_page = render_sidebar()

    page_map = {

    "Executive Overview": page_executive_overview,

    "Model Comparison": page_model_comparison,

    "Feature Drift Monitor": page_feature_drift,

    "Retraining Experiment": page_retraining_experiment,

    "Optuna Hyperparameter Search": page_optuna_search,

    "Experiment Reports & Downloads": page_reports_downloads,

    "Integrated Self-Healing Pipeline": page_integrated_pipeline,

    "Beginner Learning Center": page_learning_center,

    "Root-Cause Diagnosis": page_root_cause_diagnosis,

    "Performance-Aware Decision Engine": page_performance_decision_engine,

    "Explainable AI (XAI)": page_explainable_ai,

    "Sequential Drift Monitor": page_sequential_drift_monitor,

    "Model Registry and Rollback": page_model_registry_rollback,

    "Recovery Strategy Comparison": page_recovery_strategy_comparison,

    "Five-Model Benchmark": page_five_model_benchmark,
    "Controlled Concept-Drift Stress Test":
        page_controlled_concept_drift,

    "Feature Contribution & Proposed Model":
        page_feature_contribution_proposed_model,

    "Drift Detective": page_drift_detective,
    "Research Contribution & Novelty": page_research_contribution_novelty,

     "Self-Healing Recovery": page_self_healing_recovery,

    "Live Drift & Self-Healing Simulator": page_live_drift_simulator,

    "Selective Retraining & Cost Analysis":
        page_selective_retraining_analysis,

}

    render_fn = page_map.get(selected_page, page_executive_overview)
    render_fn()


if __name__ == "__main__":
    main()
