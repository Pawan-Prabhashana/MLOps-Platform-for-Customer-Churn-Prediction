"""MLflow tracking utilities: URI setup, experiment management, run helpers.

All code reads the tracking URI from settings / configs/model.yaml — no
hardcoded literals anywhere.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
import yaml
from mlflow import MlflowClient
from mlflow.entities import Run

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_MODEL_YAML = _REPO_ROOT / "configs" / "model.yaml"


# ── Config helpers ────────────────────────────────────────────────────────────

def load_mlflow_config() -> dict:
    with _MODEL_YAML.open() as f:
        return yaml.safe_load(f)["mlflow"]


def get_tracking_uri() -> str:
    """Return the MLflow tracking URI from config (overridable by env var)."""
    import os
    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri
    try:
        from churnops.config import get_settings
        return get_settings().mlflow_tracking_uri
    except Exception:  # noqa: BLE001
        return load_mlflow_config()["tracking_uri"]


def get_experiment_name() -> str:
    return load_mlflow_config()["experiment_name"]


def get_registered_model_name() -> str:
    return load_mlflow_config()["registered_model_name"]


def get_promotion_metric() -> str:
    return load_mlflow_config()["promotion_metric"]


# ── Setup ─────────────────────────────────────────────────────────────────────

def setup_tracking(tracking_uri: str | None = None) -> str:
    """Set the MLflow tracking URI and ensure the default experiment exists.

    Returns the resolved tracking URI.
    """
    uri = tracking_uri or get_tracking_uri()
    mlflow.set_tracking_uri(uri)

    exp_name = get_experiment_name()
    if mlflow.get_experiment_by_name(exp_name) is None:
        mlflow.create_experiment(exp_name)
        logger.info("Created MLflow experiment: %s", exp_name)
    mlflow.set_experiment(exp_name)
    logger.debug("MLflow tracking URI: %s  experiment: %s", uri, exp_name)
    return uri


# ── Git tag ───────────────────────────────────────────────────────────────────

def _git_commit_sha() -> str:
    """Return short HEAD SHA, or 'unknown' if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(_REPO_ROOT),
        )
        return result.stdout.strip() or "unknown"
    except FileNotFoundError:
        return "unknown"


# ── Run context manager ───────────────────────────────────────────────────────

@contextmanager
def start_run(
    run_name: str,
    tags: dict[str, str] | None = None,
    tracking_uri: str | None = None,
) -> Generator[Run, None, None]:
    """Context manager that opens an MLflow run with standard churnops tags.

    Usage::

        with start_run("logistic_regression") as run:
            mlflow.log_params(...)
            mlflow.log_metrics(...)
            yield run
    """
    setup_tracking(tracking_uri)

    mlflow_cfg = load_mlflow_config()
    base_tags = {
        "project": "churnops",
        "dataset_version": mlflow_cfg.get("dataset_version", "v1"),
        "git_commit": _git_commit_sha(),
    }
    if tags:
        base_tags.update(tags)

    with mlflow.start_run(run_name=run_name, tags=base_tags) as run:
        logger.info("MLflow run started: %s  (id=%s)", run_name, run.info.run_id[:8])
        yield run
        logger.info("MLflow run finished: %s", run.info.run_id[:8])


# ── Log helpers ───────────────────────────────────────────────────────────────

def log_model_params(model_key: str, model_cfg: dict, seed: int) -> None:
    """Log model hyperparameters + metadata as MLflow params."""
    params: dict[str, Any] = {
        "model_key": model_key,
        "model_class": model_cfg["class"],
        "random_seed": seed,
    }
    params.update({f"param_{k}": str(v) for k, v in model_cfg.get("params", {}).items()})
    mlflow.log_params(params)


def log_split_metrics(metrics: dict) -> None:
    """Flatten val/test metrics dicts and log with split-prefixed names."""
    flat: dict[str, float] = {}
    for split in ("val", "test"):
        split_m = metrics.get(split, {})
        for k, v in split_m.items():
            if k not in ("split", "n_samples", "confusion_matrix") and isinstance(v, (int, float)):
                flat[f"{split}_{k}"] = float(v)
    mlflow.log_metrics(flat)


def log_data_params(data_cfg: dict) -> None:
    """Log split ratios and dataset path as params."""
    splits = data_cfg.get("splits", {})
    mlflow.log_params({
        "split_train": splits.get("train", "?"),
        "split_val": splits.get("val", "?"),
        "split_test": splits.get("test", "?"),
        "raw_csv": data_cfg.get("paths", {}).get("raw_csv", "?"),
    })


def get_run_metric(run_id: str, metric_name: str, client: MlflowClient | None = None) -> float | None:
    """Fetch a scalar metric from a completed run. Returns None if not found."""
    c = client or MlflowClient()
    try:
        history = c.get_metric_history(run_id, metric_name)
        if history:
            return history[-1].value
    except Exception:  # noqa: BLE001
        pass
    return None
