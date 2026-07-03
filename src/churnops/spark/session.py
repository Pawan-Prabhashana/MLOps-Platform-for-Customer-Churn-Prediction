"""SparkSession factory + config loader for the local MLlib path.

Local mode only (master ``local[*]``) — no cluster required. The session is
cached so repeated calls in the same process reuse one JVM.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pyspark.sql import DataFrame, SparkSession

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_SPARK_YAML = _REPO_ROOT / "configs" / "spark.yaml"
_DATA_YAML = _REPO_ROOT / "configs" / "data.yaml"


def load_spark_config() -> dict[str, Any]:
    """Load and return the parsed configs/spark.yaml."""
    with _SPARK_YAML.open() as f:
        return yaml.safe_load(f)


def load_data_config() -> dict[str, Any]:
    """Load and return the parsed configs/data.yaml (shared with the sklearn path)."""
    with _DATA_YAML.open() as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def get_spark(app_name: str | None = None) -> SparkSession:
    """Return a cached local SparkSession configured for dev.

    Args:
        app_name: Override the app name (defaults to configs/spark.yaml).

    Returns:
        An active, reusable SparkSession running in ``local[*]`` mode.
    """
    cfg = load_spark_config()["session"]
    name = app_name or cfg["app_name"]

    builder = (
        SparkSession.builder.appName(name)
        .master(cfg.get("master", "local[*]"))
        .config("spark.sql.shuffle.partitions", str(cfg.get("shuffle_partitions", 8)))
        .config("spark.sql.adaptive.enabled", str(cfg.get("adaptive_enabled", True)).lower())
        # Keep the driver lightweight and avoid noisy UI on a laptop
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.driver.memory", "2g")
    )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(cfg.get("log_level", "WARN"))
    logger.info("SparkSession ready: app=%s master=%s", name, cfg.get("master", "local[*]"))
    return spark


def read_split(spark: SparkSession, split: str) -> DataFrame:
    """Read a processed parquet split ('train' | 'val' | 'test') as a Spark DF.

    Uses the exact same files the sklearn path consumes.
    """
    data_cfg = load_data_config()
    proc_dir = _REPO_ROOT / data_cfg["paths"]["processed_dir"]
    file_map = {
        "train": data_cfg["paths"]["train_file"],
        "val": data_cfg["paths"]["val_file"],
        "test": data_cfg["paths"]["test_file"],
    }
    if split not in file_map:
        raise ValueError(f"Unknown split '{split}'. Expected one of {list(file_map)}.")
    path = proc_dir / file_map[split]
    if not path.exists():
        raise FileNotFoundError(
            f"Processed split not found: {path}. Run `python pipelines/build_dataset.py` first."
        )
    return spark.read.parquet(str(path))


def stop_spark() -> None:
    """Stop the cached SparkSession and clear the cache (useful in tests)."""
    if get_spark.cache_info().currsize:  # type: ignore[attr-defined]
        spark = get_spark()
        spark.stop()
    get_spark.cache_clear()
