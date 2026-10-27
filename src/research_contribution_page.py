from pathlib import Path

import json
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_DIR = PROJECT_ROOT / "data"


def _load_csv(path: Path):
    if not path.exists():
        return None

    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    return df if not df.empty else None


def page_research_contribution_novelty() -> None:

    st.markdown(
        """
        <div class="shml-hero">
            <div class="shml-badge">
                RESEARCH CONTRIBUTION · SYSTEM-LEVEL NOVELTY
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "## RESEARCH CONTRIBUTION & NOVELTY"
    )

    st.markdown(
        "This page explains the research gap addressed by SH-DAS, "
        "the capabilities added beyond conventional classifiers, "
        "the self-healing architecture, and the experimental evidence "
        "supporting the proposed framework."
    )

# ================================================================
    # 1. RESEARCH GAP
    # ================================================================

    st.markdown(
        '<div class="shml-section-title">'
        '1. Research Gap → Proposed Contribution'
        '</div>',
        unsafe_allow_html=True,
    )

    gap_df = pd.DataFrame(
        [
            {
                "Existing Approach":
                    "Static ML classifier",
                "Typical Capability":
                    "Predicts from the training feature space",
                "Gap":
                    "Does not provide a self-healing drift-recovery loop",
                "SH-DAS Contribution":
                    "Detect → Diagnose → Adapt → Evaluate",
            },
            {
                "Existing Approach":
                    "Drift monitoring",
                "Typical Capability":
                    "Detects distribution change",
                "Gap":
                    "Detection alone does not determine whether retraining is useful",
                "SH-DAS Contribution":
                    "Feature-aware retraining decision",
            },
            {
                "Existing Approach":
                    "Full retraining",
                "Typical Capability":
                    "Retrains after a drift event",
                "Gap":
                    "Can trigger unnecessary retraining",
                "SH-DAS Contribution":
                    "Feature contribution + drift-aware selection",
            },
            {
                "Existing Approach":
                    "Model stacking",
                "Typical Capability":
                    "Combines multiple model predictions",
                "Gap":
                    "Not inherently a drift-aware self-healing workflow",
                "SH-DAS Contribution":
                    "Stacked predictions + drift-aware information",
            },
            {
                "Existing Approach":
                    "Candidate retraining",
                "Typical Capability":
                    "Produces an updated model",
                "Gap":
                    "A new candidate may be worse than the previous model",
                "SH-DAS Contribution":
                    "Performance gate before promotion",
            },
            {
                "Existing Approach":
                    "Model replacement",
                "Typical Capability":
                    "Deploys a selected model",
                "Gap":
                    "Unsafe replacement can propagate degradation",
                "SH-DAS Contribution":
                    "Registry + rollback mechanism",
            },
        ]
    )

    st.dataframe(
        gap_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        """
        <div class="shml-card">

        <strong>Research contribution:</strong><br><br>

        SH-DAS integrates feature-level drift diagnosis, multi-model
        prediction signals, adaptive recovery, performance validation,
        and rollback into one self-healing workflow.

        <br><br>

        The contribution is therefore positioned as
        <strong>system-level integration and decision logic</strong>,
        rather than claiming that individual techniques such as
        stacking, drift detection, or retraining were invented by this project.

        </div>
        """,
        unsafe_allow_html=True,
    )

    # ================================================================
    # 2. SIX-MODEL CAPABILITY MATRIX
    # ================================================================

    st.markdown(
        '<div class="shml-section-title">'
        '2. Six-Model Capability Comparison'
        '</div>',
        unsafe_allow_html=True,
    )

    capability_df = pd.DataFrame(
        [
            {
                "Capability":
                    "19 original customer features",
                "Logistic Regression": "✓",
                "Decision Tree": "✓",
                "Random Forest": "✓",
                "Gradient Boosting": "✓",
                "SVM": "✓",
                "SH-DAS": "✓",
            },
            {
                "Capability":
                    "Multiple model prediction signals",
                "Logistic Regression": "✗",
                "Decision Tree": "✗",
                "Random Forest": "✗",
                "Gradient Boosting": "✗",
                "SVM": "✗",
                "SH-DAS": "✓",
            },
            {
                "Capability":
                    "Feature-level drift information",
                "Logistic Regression": "✗",
                "Decision Tree": "✗",
                "Random Forest": "✗",
                "Gradient Boosting": "✗",
                "SVM": "✗",
                "SH-DAS": "✓",
            },
            {
                "Capability":
                    "Second-level meta-model",
                "Logistic Regression": "✗",
                "Decision Tree": "✗",
                "Random Forest": "✗",
                "Gradient Boosting": "✗",
                "SVM": "✗",
                "SH-DAS": "✓",
            },
            {
                "Capability":
                    "Feature-aware retraining decision",
                "Logistic Regression": "✗",
                "Decision Tree": "✗",
                "Random Forest": "✗",
                "Gradient Boosting": "✗",
                "SVM": "✗",
                "SH-DAS": "✓",
            },
            {
                "Capability":
                    "Performance-aware promotion gate",
                "Logistic Regression": "✗",
                "Decision Tree": "✗",
                "Random Forest": "✗",
                "Gradient Boosting": "✗",
                "SVM": "✗",
                "SH-DAS": "✓",
            },
            {
                "Capability":
                    "Rollback-safe model selection",
                "Logistic Regression": "✗",
                "Decision Tree": "✗",
                "Random Forest": "✗",
                "Gradient Boosting": "✗",
                "SVM": "✗",
                "SH-DAS": "✓",
            },
        ]
    )

    st.dataframe(
        capability_df,
        use_container_width=True,
        hide_index=True,
    )

    # ================================================================

    # ================================================================
    # 3. ARCHITECTURE
    # ================================================================

    st.markdown(
        '<div class="shml-section-title">'
        '3. SH-DAS Self-Healing Architecture'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "The proposed framework connects drift monitoring, "
        "feature-level diagnosis, multi-model prediction, adaptive "
        "recovery, performance validation, and safe model selection."
    )

    # Step 1
    st.info("1. INPUT DATA → Customer feature data")

    # Step 2
    st.info("2. FEATURE MONITORING → Monitor feature behaviour")

    # Step 3
    st.warning(
        "3. DRIFT DETECTION → Identify distribution change"
    )

    # Step 4
    st.warning(
        "4. FEATURE-LEVEL DIAGNOSIS → Identify affected features"
    )

    # Step 5
    st.markdown("### 5. FIVE CONVENTIONAL BASE MODELS")

    model_cols = st.columns(5)

    model_names = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "SVM",
    ]

    for col, model_name in zip(model_cols, model_names):
        with col:
            st.markdown(
                f"""
                <div style="
                    border:1px solid rgba(99,102,241,.25);
                    border-radius:12px;
                    padding:0.8rem;
                    text-align:center;
                    background:rgba(99,102,241,.06);
                    min-height:70px;
                ">
                    <strong>{model_name}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### 6. PREDICTION + DRIFT SIGNALS")

    signal_cols = st.columns(2)

    with signal_cols[0]:
        st.info(
            "Five base-model prediction signals"
        )

    with signal_cols[1]:
        st.warning(
            "Feature-level drift information"
        )

    # Step 7
    st.markdown("### 7. SH-DAS META-MODEL")

    st.success(
        "The meta-model combines the available prediction and "
        "drift-aware signals to produce the proposed decision."
    )

    # Step 8
    st.markdown("### 8. CANDIDATE RECOVERY MODEL")

    st.info(
        "An adaptive candidate can be generated using the recent "
        "adaptation data."
    )

    # Step 9
    st.markdown("### 9. PERFORMANCE GATE")

    gate_cols = st.columns(2)

    with gate_cols[0]:
        st.success(
            "PASS\n\nPromote Candidate"
        )

    with gate_cols[1]:
        st.error(
            "FAIL\n\nRetain / Rollback"
        )

    st.markdown(
        """
        <div class="shml-card">

        <strong>Self-Healing Logic</strong><br><br>

        Detect drift → diagnose affected features → generate adaptive
        candidate → evaluate performance → promote only when the
        candidate satisfies the decision criteria → otherwise retain
        or rollback to the previous model.

        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4. DATA LINEAGE
    # ================================================================

    st.markdown(
        '<div class="shml-section-title">'
        '4. Data &amp; Experiment Lineage'
        '</div>',
        unsafe_allow_html=True,
    )

    lineage_df = pd.DataFrame(
        [
            {
                "Stage":
                    "Original benchmark",
                "Dataset":
                    "IBM Telco Customer Churn",
                "Purpose":
                    "Baseline and conventional model evaluation",
            },
            {
                "Stage":
                    "Feature drift experiment",
                "Dataset":
                    "Controlled offline drift simulation",
                "Purpose":
                    "Study performance under distribution shift",
            },
            {
                "Stage":
                    "Controlled concept-drift stress test",
                "Dataset":
                    "Synthetic Telco-like 12,000 records",
                "Purpose":
                    "Evaluate self-healing adaptation",
            },
            {
                "Stage":
                    "Recovery decision",
                "Dataset":
                    "Adaptation window",
                "Purpose":
                    "Generate and validate an adaptive candidate",
            },
        ]
    )

    st.dataframe(
        lineage_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        """
        <div class="shml-warning">

        <strong>Experiment integrity:</strong><br><br>

        The synthetic concept-drift stress test is reported separately from
        the original IBM Telco experiment. Its results should not be described
        as additional IBM Telco observations.

        </div>
        """,
        unsafe_allow_html=True,
    )

    # ================================================================
    # 5. CONTROLLED STRESS TEST
    # ================================================================

    st.markdown(
        '<div class="shml-section-title">'
        '5. Controlled Concept-Drift Evidence'
        '</div>',
        unsafe_allow_html=True,
    )

    controlled_path = (
        REPORTS_DIR
        / "controlled_concept_drift"
        / "concept_drift_comparison.csv"
    )

    controlled_df = _load_csv(controlled_path)

    if controlled_df is not None:

        controlled_df.columns = [
            str(c).strip()
            for c in controlled_df.columns
        ]

        model_col = None

        for name in [
            "Model",
            "model",
            "System",
            "system",
            "Strategy",
            "strategy",
        ]:
            if name in controlled_df.columns:
                model_col = name
                break

        if model_col is not None:

            adaptive_mask = (
                controlled_df[model_col]
                .astype(str)
                .str.contains(
                    "SH-DAS|Adaptive",
                    case=False,
                    regex=True,
                )
            )

            frozen = controlled_df[
                ~adaptive_mask
            ]

            adaptive = controlled_df[
                adaptive_mask
            ]

            if not adaptive.empty and not frozen.empty:

                adaptive_row = adaptive.iloc[0]

                metrics = {
                    "Accuracy": "Accuracy",
                    "Precision": "Precision",
                    "Recall": "Recall",
                    "F1-score": "F1-score",
                    "ROC-AUC": "ROC-AUC",
                }

                result_rows = []

                for display_name, column in metrics.items():

                    if column not in controlled_df.columns:
                        continue

                    frozen_values = pd.to_numeric(
                        frozen[column],
                        errors="coerce",
                    ).dropna()

                    if frozen_values.empty:
                        continue

                    best_frozen = float(
                        frozen_values.max()
                    )

                    adaptive_value = float(
                        adaptive_row[column]
                    )

                    result_rows.append(
                        {
                            "Metric":
                                display_name,
                            "Best Frozen":
                                best_frozen,
                            "SH-DAS Adaptive":
                                adaptive_value,
                            "Improvement":
                                adaptive_value - best_frozen,
                        }
                    )

                stress_df = pd.DataFrame(
                    result_rows
                )

                st.dataframe(
                    stress_df,
                    use_container_width=True,
                    hide_index=True,
                )

                passes = int(
                    (
                        stress_df["Improvement"] > 0
                    ).sum()
                )

                if passes == 5:
                    st.success(
                        "Controlled stress test: "
                        "SH-DAS improved all five evaluated metrics — 5/5."
                    )
                else:
                    st.info(
                        f"Controlled stress test: "
                        f"{passes}/5 metrics improved."
                    )

    else:

        st.info(
            "Controlled concept-drift results are not available yet."
        )

    # ================================================================
    # 6. SELECTIVE RETRAINING
    # ================================================================

    st.markdown(
        '<div class="shml-section-title">'
        '6. Selective Retraining Evidence'
        '</div>',
        unsafe_allow_html=True,
    )

    selective_path = (
        REPORTS_DIR
        / "selective_retraining"
        / "selective_retraining_summary.csv"
    )

    selective_df = _load_csv(
        selective_path
    )

    if selective_df is not None:

        st.dataframe(
            selective_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "Selective retraining summary is not available."
        )

    # ================================================================
    # 7. PROPOSED MATHEMATICAL CONTRIBUTION - DRA
    # ================================================================

    st.markdown(
        '<div class="shml-section-title">'
        '7. Proposed Mathematical Contribution — '
        'Drift-Relevance Alignment (DRA)'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="shml-info">
        <strong>SH-DAS proposes a project-specific Drift-Relevance Alignment
        (DRA) score</strong> to quantify whether observed feature drift is
        concentrated on features that have meaningful measured contribution
        to the predictive model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="shml-section-sub">Existing Drift Measure vs. Proposed SH-DAS Formulation</div>',
        unsafe_allow_html=True,
    )

    comparison_col1, comparison_col2 = st.columns(2, gap="large")

    with comparison_col1:
        with st.container(border=True):
            st.markdown("### Existing Approach: Population Stability Index (PSI)")
            st.markdown(
                "**What PSI tells us:** It quantifies the magnitude of distributional change."
            )

            st.latex(
                r"PSI = \sum_{i=1}^{B}(P_i-Q_i)"
                r"\ln\left(\frac{P_i}{Q_i}\right)"
            )

            st.markdown(
                "**Limitation for SH-DAS:** PSI by itself does not incorporate "
                "the measured contribution of each feature to the predictive model."
            )

    with comparison_col2:
        with st.container(border=True):
            st.markdown("### Proposed SH-DAS: Drift-Relevance Alignment (DRA)")
            st.markdown(
                "**Core idea:** DRA combines normalized feature drift with "
                "measured model-relevant feature contribution."
            )

            st.latex(
                r"DRA_t = \sum_{j=1}^{d} w_j d_{j,t}"
            )

            st.markdown("**Normalized drift intensity:**")
            st.latex(
                r"d_{j,t} = "
                r"\min\left(\frac{PSI_{j,t}}{\tau_{high}},1\right)"
            )

            st.markdown("**Contribution weight:**")
            st.latex(
                r"w_j = "
                r"\frac{\max(I_j,0)}{\sum_k\max(I_k,0)}"
            )

            st.markdown(
                "**What DRA adds:** It connects distributional drift with "
                "measured model-relevant feature contribution."
            )

    st.markdown(
        """
        <div class="shml-card">

        <h3>Meaning of the Terms</h3>

        <table style="width:100%; border-collapse:collapse;">
        <tr>
            <th style="text-align:left; padding:8px;">Term</th>
            <th style="text-align:left; padding:8px;">Meaning</th>
        </tr>
        <tr>
            <td style="padding:8px;"><strong>PSI</strong></td>
            <td style="padding:8px;">
                Population Stability Index measuring distributional drift.
            </td>
        </tr>
        <tr>
            <td style="padding:8px;"><strong>d<sub>j,t</sub></strong></td>
            <td style="padding:8px;">
                Normalized drift intensity for feature j.
            </td>
        </tr>
        <tr>
            <td style="padding:8px;"><strong>I<sub>j</sub></strong></td>
            <td style="padding:8px;">
                Measured permutation importance of feature j.
            </td>
        </tr>
        <tr>
            <td style="padding:8px;"><strong>w<sub>j</sub></strong></td>
            <td style="padding:8px;">
                Normalized contribution weight of feature j.
            </td>
        </tr>
        <tr>
            <td style="padding:8px;"><strong>DRA</strong></td>
            <td style="padding:8px;">
                Overall alignment between observed feature drift and measured
                model-relevant feature contribution.
            </td>
        </tr>
        </table>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="shml-section-sub">'
        'Controlled experimental evidence'
        '</div>',
        unsafe_allow_html=True,
    )

    dra_summary_path = (
        REPORTS_DIR
        / "selective_retraining"
        / "dra_experiment_summary.csv"
    )

    dra_summary_df = _load_csv(
        dra_summary_path
    )

    if dra_summary_df is not None:

        st.dataframe(
            dra_summary_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "DRA experiment summary is not available yet."
        )

    st.markdown(
        """
        <div class="shml-card">

        <h3>Key Controlled Observation</h3>

        <p>
        The controlled experiment produced two important cases with the same
        maximum PSI of approximately <strong>1.4731</strong>, but substantially
        different DRA scores.
        </p>

        <table style="width:100%; border-collapse:collapse;">
        <tr>
            <th style="text-align:left; padding:8px;">Scenario</th>
            <th style="text-align:left; padding:8px;">Max PSI</th>
            <th style="text-align:left; padding:8px;">DRA</th>
            <th style="text-align:left; padding:8px;">DRA Trigger</th>
        </tr>
        <tr>
            <td style="padding:8px;">
                Strong MonthlyCharges drift
            </td>
            <td style="padding:8px;">1.4731</td>
            <td style="padding:8px;">0.2197</td>
            <td style="padding:8px;">False</td>
        </tr>
        <tr>
            <td style="padding:8px;">
                Strong important-feature drift
            </td>
            <td style="padding:8px;">1.4731</td>
            <td style="padding:8px;">0.9307</td>
            <td style="padding:8px;">True</td>
        </tr>
        </table>

        <p style="margin-top:1rem;">
        This demonstrates that maximum PSI alone does not describe where the
        detected drift is concentrated. DRA adds a feature-contribution
        dimension to the drift signal.
        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "Research positioning: DRA is presented as a proposed "
        "project-specific mathematical formulation. The controlled "
        "experiment demonstrates a distinct feature-aware signal; further "
        "policy-level evaluation is required before claiming that DRA "
        "universally improves retraining decisions."
    )

    # ================================================================
    # 8. ABLATION STUDY
    # ================================================================


    st.markdown(
        '<div class="shml-section-title">'
        '7. Ablation Study'
        '</div>',
        unsafe_allow_html=True,
    )

    ablation_path = (
        REPORTS_DIR
        / "ablation_study"
        / "ablation_results.csv"
    )

    ablation_df = _load_csv(
        ablation_path
    )

    if ablation_df is not None:

        st.dataframe(
            ablation_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "Ablation results have not been generated yet. "
            "The dashboard section is ready for the experiment artifact."
        )

    # ================================================================
    # 8. ROBUSTNESS
    # ================================================================

    st.markdown(
        '<div class="shml-section-title">'
        '8. Multi-Seed Robustness'
        '</div>',
        unsafe_allow_html=True,
    )

    robustness_candidates = [
        REPORTS_DIR / "multi_seed_robustness.csv",
        REPORTS_DIR / "multi_seed_robustness" / "robustness_results.csv",
        REPORTS_DIR / "robustness" / "multi_seed_results.csv",
    ]

    robustness_df = None

    for candidate in robustness_candidates:

        robustness_df = _load_csv(candidate)

        if robustness_df is not None:
            break

    if robustness_df is not None:

        st.dataframe(
            robustness_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "Multi-seed robustness artifact has not been generated yet."
        )

    # ================================================================
    # FINAL RESEARCH POSITIONING
    # ================================================================

    st.markdown(
        '<div class="shml-section-title">'
        '9. Final Research Positioning'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="shml-card">

        <strong>What we claim:</strong><br><br>

        SH-DAS provides a unified framework that connects feature-level
        drift diagnosis with selective retraining, adaptive recovery,
        performance validation, and rollback-safe model management.

        <br><br>

        <strong>What we do not claim:</strong><br><br>

        We do not claim that stacking, drift detection, retraining,
        permutation importance, or rollback were individually invented
        by this project.

        <br><br>

        <strong>Research value:</strong><br><br>

        The project evaluates how these components can operate together
        as a self-healing ML decision process, including controlled
        concept-drift testing and safeguards against accepting degraded
        recovery candidates.

        </div>
        """,
        unsafe_allow_html=True,
    )


