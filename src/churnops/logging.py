"""Logging setup helper for churnops.

Call `setup_logging()` once at process startup (e.g., in a pipeline entrypoint)
to configure the root logger from configs/logging.yaml.
"""

from __future__ import annotations

import logging
import logging.config
import logging.handlers  # noqa: F401 – needed for RotatingFileHandler via dictConfig
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).parent.parent.parent
_LOG_YAML = _REPO_ROOT / "configs" / "logging.yaml"


def setup_logging(config_path: Path | None = None) -> None:
    """Configure logging from a YAML file (defaults to configs/logging.yaml).

    Creates the logs/ directory if the file handler references it and it
    does not yet exist.
    """
    path = Path(config_path) if config_path else _LOG_YAML
    if not path.exists():
        logging.basicConfig(level=logging.INFO)
        logging.getLogger(__name__).warning(
            "Logging config not found at %s; using basicConfig.", path
        )
        return

    with path.open() as f:
        cfg = yaml.safe_load(f)

    # Ensure log directory exists for any file handlers
    for handler_cfg in (cfg.get("handlers") or {}).values():
        filename = handler_cfg.get("filename")
        if filename:
            Path(filename).parent.mkdir(parents=True, exist_ok=True)

    logging.config.dictConfig(cfg)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Call setup_logging() first."""
    return logging.getLogger(name)
