"""Evaluation plots for MLflow artifacts.

Produces and saves PNGs for:
  - Confusion matrix (with counts + percentages)
  - ROC curve (with AUC annotation)
  - Precision-Recall curve (with AP annotation)
  - Feature importance / coefficient plot (when the estimator supports it,
    with feature names recovered from the ColumnTransformer)

All functions return a Path to the saved PNG so callers can log them
as MLflow artifacts.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline

matplotlib.use("Agg")  # non-interactive backend — safe in headless / Docker envs

logger = logging.getLogger(__name__)


def _fig_path(out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{name}.png"


# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    out_dir: Path,
    title: str = "Confusion Matrix",
) -> Path:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
    fig.tight_layout()
    path = _fig_path(out_dir, "confusion_matrix")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.debug("Saved confusion matrix → %s", path)
    return path


# ── ROC curve ─────────────────────────────────────────────────────────────────

def plot_roc_curve(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    out_dir: Path,
    title: str = "ROC Curve",
) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax, name="Churn model")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = _fig_path(out_dir, "roc_curve")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.debug("Saved ROC curve → %s", path)
    return path


# ── Precision-Recall curve ────────────────────────────────────────────────────

def plot_pr_curve(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    out_dir: Path,
    title: str = "Precision-Recall Curve",
) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax, name="Churn model")
    prevalence = float(np.mean(y_true))
    ax.axhline(prevalence, color="gray", linestyle="--", lw=0.8, label=f"Baseline ({prevalence:.2f})")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = _fig_path(out_dir, "pr_curve")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.debug("Saved PR curve → %s", path)
    return path


# ── Feature importance / coefficients ────────────────────────────────────────

def _get_feature_names(pipeline: Pipeline) -> list[str] | None:
    """Extract feature names from the ColumnTransformer step."""
    try:
        preproc = pipeline.named_steps["preprocess"]
        return list(preproc.get_feature_names_out())
    except Exception:  # noqa: BLE001
        return None


def _get_importances(pipeline: Pipeline) -> tuple[np.ndarray | None, list[str] | None]:
    """Return (importances, feature_names) if the estimator supports it."""
    estimator = pipeline.named_steps["model"]
    names = _get_feature_names(pipeline)

    if hasattr(estimator, "feature_importances_"):
        return estimator.feature_importances_, names
    if hasattr(estimator, "coef_"):
        coef = estimator.coef_
        importances = np.abs(coef[0]) if coef.ndim == 2 else np.abs(coef)
        return importances, names
    return None, names


def plot_feature_importance(
    pipeline: Pipeline,
    out_dir: Path,
    top_n: int = 20,
    title: str = "Top Feature Importances",
) -> Path | None:
    """Save a horizontal bar chart of the top-N most important features.

    Returns None (and skips the plot) if the estimator doesn't expose
    feature_importances_ or coef_.
    """
    importances, names = _get_importances(pipeline)
    if importances is None or names is None:
        logger.debug("Estimator does not expose feature importances — skipping plot.")
        return None

    importances = np.array(importances)
    names = list(names)
    if len(importances) != len(names):
        logger.warning("Importance / name length mismatch — skipping feature importance plot.")
        return None

    # Take top-N by magnitude
    idx = np.argsort(importances)[-top_n:]
    top_names = [names[i] for i in idx]
    top_vals = importances[idx]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    ax.barh(range(len(top_names)), top_vals, color="steelblue")
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=8)
    ax.set_xlabel("Importance (absolute value)")
    ax.set_title(title)
    fig.tight_layout()
    path = _fig_path(out_dir, "feature_importance")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.debug("Saved feature importance → %s", path)
    return path


# ── All-in-one ────────────────────────────────────────────────────────────────

def generate_all_plots(
    pipeline: Pipeline,
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    out_dir: Path,
    split_name: str = "test",
    model_key: str = "",
) -> list[Path]:
    """Generate all evaluation plots and return a list of created file paths."""
    prefix = f"{model_key}_{split_name}" if model_key else split_name
    paths: list[Path] = []

    paths.append(
        plot_confusion_matrix(y_true, y_pred, out_dir, title=f"Confusion Matrix — {prefix}")
    )
    paths.append(
        plot_roc_curve(y_true, y_proba, out_dir, title=f"ROC Curve — {prefix}")
    )
    paths.append(
        plot_pr_curve(y_true, y_proba, out_dir, title=f"PR Curve — {prefix}")
    )
    fi_path = plot_feature_importance(
        pipeline, out_dir, title=f"Feature Importance — {prefix}"
    )
    if fi_path:
        paths.append(fi_path)

    return paths
