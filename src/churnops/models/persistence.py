"""Save and load sklearn pipelines via joblib.

Each saved artifact consists of two files:
  - <pipeline_file>   — the serialised sklearn Pipeline (joblib)
  - <sidecar_file>    — a JSON with model name, metrics, schema version, timestamp
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import yaml
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_MODEL_YAML = _REPO_ROOT / "configs" / "model.yaml"

# Schema version bumped whenever column schema changes break compatibility
SCHEMA_VERSION = "1.0.0"


def _default_paths() -> tuple[Path, Path]:
    with _MODEL_YAML.open() as f:
        cfg = yaml.safe_load(f)["artifacts"]
    base = _REPO_ROOT / cfg["base_dir"]
    return base / cfg["pipeline_file"], base / cfg["sidecar_file"]


def save_pipeline(
    pipeline: Pipeline,
    metrics: dict,
    pipeline_path: Path | str | None = None,
    sidecar_path: Path | str | None = None,
) -> tuple[Path, Path]:
    """Persist a fitted Pipeline and a descriptive sidecar JSON.

    Args:
        pipeline:      Fitted sklearn Pipeline.
        metrics:       Metrics dict from train.train().
        pipeline_path: Override for the .joblib path (default from config).
        sidecar_path:  Override for the JSON sidecar (default from config).

    Returns:
        (pipeline_path, sidecar_path)
    """
    default_pipe, default_side = _default_paths()
    pipeline_path = Path(pipeline_path) if pipeline_path else default_pipe
    sidecar_path = Path(sidecar_path) if sidecar_path else default_side

    pipeline_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, pipeline_path)
    logger.info("Pipeline saved → %s", pipeline_path)

    sidecar: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "saved_at": datetime.now(UTC).isoformat(),
        "model_key": metrics.get("model_key"),
        "model_class": metrics.get("model_class"),
        "val_metrics": {
            k: v
            for k, v in metrics.get("val", {}).items()
            if k != "confusion_matrix"
        },
        "test_metrics": {
            k: v
            for k, v in metrics.get("test", {}).items()
            if k != "confusion_matrix"
        },
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    logger.info("Sidecar saved  → %s", sidecar_path)

    return pipeline_path, sidecar_path


def load_pipeline(pipeline_path: Path | str | None = None) -> Pipeline:
    """Load and return a fitted sklearn Pipeline from disk.

    Args:
        pipeline_path: Path to the .joblib file.
                       Defaults to the path from configs/model.yaml.
    """
    if pipeline_path is None:
        pipeline_path, _ = _default_paths()
    pipeline_path = Path(pipeline_path)

    if not pipeline_path.exists():
        raise FileNotFoundError(f"Pipeline artifact not found: {pipeline_path}")

    pipeline = joblib.load(pipeline_path)
    logger.info("Pipeline loaded ← %s", pipeline_path)
    return pipeline


def load_sidecar(sidecar_path: Path | str | None = None) -> dict:
    """Load and return the sidecar JSON metadata."""
    if sidecar_path is None:
        _, sidecar_path = _default_paths()
    sidecar_path = Path(sidecar_path)
    if not sidecar_path.exists():
        raise FileNotFoundError(f"Sidecar not found: {sidecar_path}")
    return json.loads(sidecar_path.read_text())
