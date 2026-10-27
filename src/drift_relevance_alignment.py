"""
SH-DAS Drift-Relevance Alignment

Project-specific mathematical formulation:

DRA_t = sum_j (w_j * d_jt)

where:
    d_jt = min(PSI_jt / PSI_high, 1)
    w_j  = max(I_j, 0) / sum_k max(I_k, 0)

The score measures whether detected feature drift is concentrated
on features that have meaningful measured contribution to the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_dra(
    drift_table: pd.DataFrame,
    contribution_table: pd.DataFrame,
    high_drift_threshold: float = 0.25,
) -> tuple[float, pd.DataFrame]:
    """
    Compute Drift-Relevance Alignment (DRA).

    Parameters
    ----------
    drift_table:
        DataFrame containing:
            Feature
            PSI

    contribution_table:
        DataFrame containing:
            Feature
            Mean_ROC_AUC_Permutation_Importance

    high_drift_threshold:
        Project high-drift PSI threshold.

    Returns
    -------
    dra_score:
        Scalar DRA score in approximately [0, 1].

    details:
        Per-feature calculation table.
    """

    required_drift = {"Feature", "PSI"}
    required_contribution = {
        "Feature",
        "Mean_ROC_AUC_Permutation_Importance",
    }

    if not required_drift.issubset(drift_table.columns):
        raise ValueError(
            "drift_table must contain Feature and PSI columns."
        )

    if not required_contribution.issubset(
        contribution_table.columns
    ):
        raise ValueError(
            "contribution_table must contain Feature and "
            "Mean_ROC_AUC_Permutation_Importance columns."
        )

    merged = drift_table[
        ["Feature", "PSI"]
    ].merge(
        contribution_table[
            [
                "Feature",
                "Mean_ROC_AUC_Permutation_Importance",
            ]
        ],
        on="Feature",
        how="inner",
    )

    if merged.empty:
        return 0.0, merged

    merged["PSI"] = pd.to_numeric(
        merged["PSI"],
        errors="coerce",
    ).fillna(0.0)

    merged[
        "Mean_ROC_AUC_Permutation_Importance"
    ] = pd.to_numeric(
        merged[
            "Mean_ROC_AUC_Permutation_Importance"
        ],
        errors="coerce",
    ).fillna(0.0)

    # ------------------------------------------------------------
    # Normalized drift intensity
    # ------------------------------------------------------------

    threshold = max(
        float(high_drift_threshold),
        1e-12,
    )

    merged["Drift_Intensity"] = np.clip(
        merged["PSI"] / threshold,
        0.0,
        1.0,
    )

    # ------------------------------------------------------------
    # Contribution weights
    # ------------------------------------------------------------

    positive_importance = np.maximum(
        merged[
            "Mean_ROC_AUC_Permutation_Importance"
        ].to_numpy(dtype=float),
        0.0,
    )

    total_importance = positive_importance.sum()

    if total_importance > 0:

        merged["Contribution_Weight"] = (
            positive_importance
            / total_importance
        )

    else:

        # Safe fallback if no positive measured
        # contribution exists.
        merged["Contribution_Weight"] = (
            1.0 / len(merged)
        )

    # ------------------------------------------------------------
    # Per-feature alignment contribution
    # ------------------------------------------------------------

    merged["Alignment_Contribution"] = (
        merged["Drift_Intensity"]
        * merged["Contribution_Weight"]
    )

    # ------------------------------------------------------------
    # Final mathematical score
    # ------------------------------------------------------------

    dra_score = float(
        merged[
            "Alignment_Contribution"
        ].sum()
    )

    return dra_score, merged.sort_values(
        "Alignment_Contribution",
        ascending=False,
    ).reset_index(drop=True)