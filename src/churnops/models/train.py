"""Training, evaluation, and model-selection logic.

Supports three estimators selectable from configs/model.yaml:
  - logistic_regression
  - random_forest
  - gradient_boosting  (HistGradientBoostingClassifier)

All randomness is seeded from settings.random_seed for reproducibility.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from churnops.config import get_settings
from churnops.data.schema import TARGET_COL
from churnops.models.pipeline import build_pipeline

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_MODEL_YAML = _REPO_ROOT / "configs" / "model.yaml"
_DATA_YAML = _REPO_ROOT / "configs" / "data.yaml"


def _load_model_config() -> dict:
    with _MODEL_YAML.open() as f:
        return yaml.safe_load(f)


def _load_data_config() -> dict:
    with _DATA_YAML.open() as f:
        return yaml.safe_load(f)


def _build_estimator(model_key: str, cfg: dict) -> Any:
    """Instantiate the estimator from its dotted class path + params."""
    model_cfg = cfg["models"][model_key]
    module_path, cls_name = model_cfg["class"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), cls_name)
    params = dict(model_cfg.get("params", {}))

    # Ensure random_state is pinned to settings if not explicitly set
    settings = get_settings()
    if "random_state" in params:
        params["random_state"] = settings.random_seed

    return cls(**params)


def _evaluate(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, split_name: str) -> dict:
    """Compute all evaluation metrics for one split."""
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]

    cm = confusion_matrix(y, y_pred)
    metrics = {
        "split": split_name,
        "n_samples": len(y),
        "accuracy": round(float(accuracy_score(y, y_pred)), 4),
        "precision": round(float(precision_score(y, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y, y_proba)), 4),
        "confusion_matrix": cm.tolist(),
    }
    logger.info(
        "[%s] roc_auc=%.4f  pr_auc=%.4f  recall=%.4f  f1=%.4f",
        split_name.upper(),
        metrics["roc_auc"],
        metrics["pr_auc"],
        metrics["recall"],
        metrics["f1"],
    )
    return metrics


def _xy(df: pd.DataFrame):
    """Split DataFrame into features (X) and target (y)."""
    return df.drop(columns=[TARGET_COL]), df[TARGET_COL]


def train(model_key: str | None = None) -> tuple[Pipeline, dict]:
    """Train the chosen model on the processed train split, evaluate on val/test.

    Args:
        model_key: Key from configs/model.yaml `models` section.
                   Defaults to `default_model` in config.

    Returns:
        Tuple of (fitted_pipeline, metrics_dict).
    """
    cfg = _load_model_config()
    data_cfg = _load_data_config()
    if model_key is None:
        model_key = cfg["default_model"]

    if model_key not in cfg["models"]:
        raise ValueError(
            f"Unknown model key '{model_key}'. "
            f"Available: {list(cfg['models'].keys())}"
        )

    proc_dir = _REPO_ROOT / data_cfg["paths"]["processed_dir"]
    train_df = pd.read_parquet(proc_dir / data_cfg["paths"]["train_file"])
    val_df = pd.read_parquet(proc_dir / data_cfg["paths"]["val_file"])
    test_df = pd.read_parquet(proc_dir / data_cfg["paths"]["test_file"])

    X_train, y_train = _xy(train_df)
    X_val, y_val = _xy(val_df)
    X_test, y_test = _xy(test_df)

    estimator = _build_estimator(model_key, cfg)
    pipeline = build_pipeline(estimator)

    logger.info("Fitting %s on %d training rows...", model_key, len(X_train))

    # HistGradientBoosting doesn't support class_weight; pass sample_weight instead
    fit_params: dict = {}
    if "HistGradient" in type(estimator).__name__:
        sw = compute_sample_weight("balanced", y_train)
        fit_params["model__sample_weight"] = sw

    pipeline.fit(X_train, y_train, **fit_params)
    logger.info("Training complete.")

    val_metrics = _evaluate(pipeline, X_val, y_val, "val")
    test_metrics = _evaluate(pipeline, X_test, y_test, "test")

    metrics = {
        "model_key": model_key,
        "model_class": cfg["models"][model_key]["class"],
        "val": val_metrics,
        "test": test_metrics,
    }
    return pipeline, metrics


def metrics_table(metrics: dict) -> str:
    """Return a formatted string table of val + test metrics."""
    header = f"\n{'Metric':<14} {'Val':>10} {'Test':>10}"
    sep = "-" * 38
    rows = [header, sep]
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    for k in keys:
        v = metrics["val"].get(k, "-")
        t = metrics["test"].get(k, "-")
        rows.append(f"{k:<14} {v:>10} {t:>10}")
    rows.append(sep)
    # confusion matrix
    cm_v = np.array(metrics["val"]["confusion_matrix"])
    cm_t = np.array(metrics["test"]["confusion_matrix"])
    rows.append(f"\nVal  confusion matrix:\n{cm_v}")
    rows.append(f"\nTest confusion matrix:\n{cm_t}")
    return "\n".join(rows)
