"""Data drift + prediction drift computation.

Pure functions only — no I/O, no Kafka, no DB. Callers (collector.py) fetch
the reference/current data and hand it here as arrays/Series/DataFrames.

What's implemented, and why
----------------------------
Prediction drift (always runs): compares the distribution of
``churn_probability`` (PSI + KS) and the predicted-label ("Yes") rate in the
current window against a reference — the current production model's scored
predictions on the TRAINING set. This always works because it only needs the
`predictions` table / a Kafka window of prediction messages, both of which
always carry churn_probability.

Data drift (runs only when raw feature values are available): per-numeric
-feature PSI and per-categorical-feature chi-square/PSI against the training
split. The ``predictions`` table does NOT store raw feature values (see
monitoring/db.py docstring) — the fixed prediction message contract is
``{customerID, churn_probability, prediction, event_ts, processed_ts,
actual_churn}``. So the "current window" for data drift is obtained
separately, by collector.py re-consuming a bounded window of the raw
``telco.raw.customers`` topic. When that window can't be obtained (Kafka
down, or ``--source db``), data drift is skipped and this is logged/reported
explicitly rather than silently guessed at.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

_EPS = 1e-6


# ── PSI (population stability index) ──────────────────────────────────────────

def psi_band(value: float, moderate: float = 0.10, significant: float = 0.25) -> str:
    """Return the standard PSI band label for a computed PSI value."""
    if value < moderate:
        return "none"
    if value < significant:
        return "moderate"
    return "significant"


def population_stability_index(
    reference: np.ndarray | pd.Series,
    current: np.ndarray | pd.Series,
    buckets: int = 10,
) -> float:
    """PSI for a numeric distribution, bucketed on reference-set quantiles."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    breakpoints = np.unique(np.quantile(ref, np.linspace(0, 1, buckets + 1)))
    if len(breakpoints) < 3:
        # Reference distribution is (near-)constant — not enough distinct
        # values to bucket meaningfully.
        return 0.0
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=breakpoints)
    cur_counts, _ = np.histogram(cur, bins=breakpoints)
    ref_pct = np.clip(ref_counts / len(ref), _EPS, None)
    cur_pct = np.clip(cur_counts / len(cur), _EPS, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    """PSI for a categorical column, bucketed on the union of observed categories."""
    ref_counts = reference.value_counts(normalize=False)
    cur_counts = current.value_counts(normalize=False)
    categories = sorted(set(ref_counts.index) | set(cur_counts.index))
    if not categories:
        return 0.0

    ref_total = max(len(reference), 1)
    cur_total = max(len(current), 1)
    ref_pct = np.clip(
        np.array([ref_counts.get(c, 0) for c in categories]) / ref_total, _EPS, None
    )
    cur_pct = np.clip(
        np.array([cur_counts.get(c, 0) for c in categories]) / cur_total, _EPS, None
    )
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks_test(reference: np.ndarray | pd.Series, current: np.ndarray | pd.Series) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test statistic and p-value."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if len(ref) < 2 or len(cur) < 2:
        return 0.0, 1.0
    result = stats.ks_2samp(ref, cur)
    return float(result.statistic), float(result.pvalue)


def chi_square_test(reference: pd.Series, current: pd.Series) -> tuple[float | None, float | None]:
    """Chi-square test of independence over category frequency tables.

    Returns (None, None) when the contingency table is degenerate (e.g. a
    category with zero expected count), rather than letting scipy raise.
    """
    categories = sorted(set(reference.dropna()) | set(current.dropna()))
    if len(categories) < 2:
        return None, None
    ref_counts = reference.value_counts()
    cur_counts = current.value_counts()
    table = np.array(
        [[ref_counts.get(c, 0) for c in categories], [cur_counts.get(c, 0) for c in categories]]
    )
    if (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any():
        return None, None
    try:
        chi2, p, _, _ = stats.chi2_contingency(table)
        return float(chi2), float(p)
    except ValueError:
        return None, None


# ── Prediction drift ──────────────────────────────────────────────────────────

def prediction_drift_report(
    reference_probs: np.ndarray | pd.Series,
    current_probs: np.ndarray | pd.Series,
    *,
    moderate_threshold: float = 0.10,
    significant_threshold: float = 0.25,
    min_sample_size: int = 30,
    histogram_buckets: int = 10,
    reference_window: str = "training-set predicted distribution",
    current_window: str | None = None,
) -> dict[str, Any]:
    """Compare churn_probability distributions; return rows ready for drift_metrics.

    Always runs (prediction_probability is always available for scored
    records) — but is skipped gracefully when the current window is too
    small to trust.
    """
    n_ref, n_cur = len(reference_probs), len(current_probs)
    current_window = current_window or f"last {n_cur} predictions"

    if n_cur < min_sample_size:
        return {
            "skipped": True,
            "reason": f"current window has {n_cur} records (< min_sample_size={min_sample_size})",
            "rows": [],
            "overall_drifted": False,
        }

    psi = population_stability_index(reference_probs, current_probs, buckets=histogram_buckets)
    ks_stat, ks_p = ks_test(reference_probs, current_probs)
    band = psi_band(psi, moderate_threshold, significant_threshold)
    is_drifted = psi >= significant_threshold

    rows = [
        {
            "feature_name": None,
            "drift_type": "prediction",
            "statistic_name": "psi",
            "statistic_value": round(psi, 6),
            "threshold": significant_threshold,
            "is_drifted": is_drifted,
            "reference_window": reference_window,
            "current_window": current_window,
        },
        {
            "feature_name": None,
            "drift_type": "prediction",
            "statistic_name": "ks",
            "statistic_value": round(ks_stat, 6),
            "threshold": 0.05,  # ks_alpha; is_drifted below is p-value based
            "is_drifted": ks_p < 0.05,
            "reference_window": reference_window,
            "current_window": current_window,
        },
    ]

    logger.info(
        "Prediction drift: psi=%.4f (%s) ks=%.4f p=%.4f n_ref=%d n_cur=%d",
        psi, band, ks_stat, ks_p, n_ref, n_cur,
    )

    return {
        "skipped": False,
        "reason": None,
        "rows": rows,
        "psi": psi,
        "psi_band": band,
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_p,
        "overall_drifted": is_drifted,
    }


# ── Data drift ────────────────────────────────────────────────────────────────

def data_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    *,
    numeric_cols: list[str],
    categorical_cols: list[str],
    psi_threshold: float = 0.25,
    chi_square_alpha: float = 0.05,
    min_sample_size: int = 30,
    histogram_buckets: int = 10,
    reference_window: str = "training split",
    current_window: str | None = None,
) -> dict[str, Any]:
    """Per-feature PSI (numeric) / chi-square + PSI (categorical) vs training data.

    Requires raw feature columns in both frames (see module docstring for why
    this needs a separately-fetched raw window). Skips gracefully when the
    current window is too small.
    """
    n_cur = len(current_df)
    current_window = current_window or f"last {n_cur} raw events"

    if n_cur < min_sample_size:
        return {
            "skipped": True,
            "reason": f"current window has {n_cur} records (< min_sample_size={min_sample_size})",
            "rows": [],
            "overall_drifted": False,
        }

    rows: list[dict[str, Any]] = []
    any_drifted = False

    for col in numeric_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        psi = population_stability_index(
            reference_df[col], current_df[col], buckets=histogram_buckets
        )
        is_drifted = psi >= psi_threshold
        any_drifted = any_drifted or is_drifted
        rows.append({
            "feature_name": col,
            "drift_type": "data",
            "statistic_name": "psi",
            "statistic_value": round(psi, 6),
            "threshold": psi_threshold,
            "is_drifted": is_drifted,
            "reference_window": reference_window,
            "current_window": current_window,
        })

    for col in categorical_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        psi = categorical_psi(reference_df[col], current_df[col])
        is_drifted = psi >= psi_threshold
        any_drifted = any_drifted or is_drifted
        rows.append({
            "feature_name": col,
            "drift_type": "data",
            "statistic_name": "psi",
            "statistic_value": round(psi, 6),
            "threshold": psi_threshold,
            "is_drifted": is_drifted,
            "reference_window": reference_window,
            "current_window": current_window,
        })

        chi2, p = chi_square_test(reference_df[col], current_df[col])
        if chi2 is not None:
            drifted_chi = p < chi_square_alpha
            any_drifted = any_drifted or drifted_chi
            rows.append({
                "feature_name": col,
                "drift_type": "data",
                "statistic_name": "chi_square",
                "statistic_value": round(chi2, 6),
                "threshold": chi_square_alpha,
                "is_drifted": drifted_chi,
                "reference_window": reference_window,
                "current_window": current_window,
            })

    logger.info(
        "Data drift: %d statistics computed across %d numeric + %d categorical features, any_drifted=%s",
        len(rows), len(numeric_cols), len(categorical_cols), any_drifted,
    )

    return {
        "skipped": False,
        "reason": None,
        "rows": rows,
        "overall_drifted": any_drifted,
    }
