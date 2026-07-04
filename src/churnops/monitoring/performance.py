"""Performance tracking from ground-truth labels carried by scored predictions.

The Kafka producer carries the real `Churn` field; the consumer forwards it
as `actual_churn` when `include_ground_truth` is enabled (see
configs/kafka.yaml). Only the subset of predictions that has a non-null
`actual_churn` contributes to these metrics — everything else is silently
excluded (not an error; most production traffic has no immediate label).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

logger = logging.getLogger(__name__)


def compute_performance_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    y_pred_label: np.ndarray | pd.Series,
) -> dict[str, float | None]:
    """Compute accuracy/precision/recall/f1/roc_auc from labeled predictions.

    `y_true` and `y_pred_label` are expected to already be 0/1 ints.
    ROC-AUC is None when the window contains only one class (undefined).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred_label)
    y_prob = np.asarray(y_prob, dtype=float)

    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if len(np.unique(y_true)) < 2:
        metrics["roc_auc"] = None
        logger.warning("roc_auc undefined: window contains only one class of actual_churn.")
    else:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))

    return metrics


def evaluate_degradation(
    current_roc_auc: float | None,
    baseline_roc_auc: float,
    tolerance: float = 0.05,
) -> tuple[bool, str]:
    """Flag performance_degraded when current roc_auc < baseline - tolerance."""
    if current_roc_auc is None:
        return False, "roc_auc unavailable this window (single-class labels) — cannot evaluate degradation."

    floor = baseline_roc_auc - tolerance
    degraded = current_roc_auc < floor
    reason = (
        f"roc_auc={current_roc_auc:.4f} vs baseline={baseline_roc_auc:.4f} "
        f"(floor={floor:.4f}, tolerance={tolerance:.4f}) → "
        f"{'DEGRADED' if degraded else 'ok'}"
    )
    return degraded, reason


def performance_report(
    df: pd.DataFrame,
    *,
    baseline_roc_auc: float,
    degradation_tolerance: float = 0.05,
    min_labeled_sample: int = 20,
    model_version: str | None = None,
) -> dict[str, Any]:
    """Full performance evaluation over a window of predictions.

    `df` must have columns: actual_churn, churn_probability, prediction.
    Rows with a null actual_churn are dropped before computing metrics.
    Gracefully reports `has_ground_truth=False` (and still lets drift run)
    when the window carries no labels at all.
    """
    labeled = df.dropna(subset=["actual_churn"]).copy()
    n = len(labeled)

    if n < min_labeled_sample:
        return {
            "has_ground_truth": n > 0,
            "sample_size": n,
            "metrics": {},
            "degraded": False,
            "reason": (
                f"only {n} labeled records this window "
                f"(< min_labeled_sample={min_labeled_sample}) — skipping performance metrics"
            ),
            "rows": [],
        }

    y_true = labeled["actual_churn"].astype(int).to_numpy()
    y_prob = labeled["churn_probability"].astype(float).to_numpy()
    y_pred = (labeled["prediction"] == "Yes").astype(int).to_numpy()

    metrics = compute_performance_metrics(y_true, y_prob, y_pred)
    degraded, reason = evaluate_degradation(metrics["roc_auc"], baseline_roc_auc, degradation_tolerance)

    window_start = labeled["processed_ts"].min() if "processed_ts" in labeled else None
    window_end = labeled["processed_ts"].max() if "processed_ts" in labeled else None

    rows = [
        {
            "metric_name": name,
            "metric_value": value,
            "sample_size": n,
            "window_start": window_start,
            "window_end": window_end,
            "model_version": model_version,
        }
        for name, value in metrics.items()
        if value is not None
    ]

    logger.info("Performance: n=%d metrics=%s degraded=%s", n, metrics, degraded)

    return {
        "has_ground_truth": True,
        "sample_size": n,
        "metrics": metrics,
        "degraded": degraded,
        "reason": reason,
        "rows": rows,
        "baseline_roc_auc": baseline_roc_auc,
        "degradation_tolerance": degradation_tolerance,
    }
