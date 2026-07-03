# ruff: noqa: E402
"""train_spark.py — Train MLlib model(s), log to MLflow, persist native Spark model.

Logs to a DISTINCT MLflow experiment ("churn-spark") and registers under a
Spark-specific model name ("churn-classifier-spark") so it never collides with
the sklearn path.

Usage
-----
    # Train all models in configs/spark.yaml
    python pipelines/train_spark.py

    # Train one estimator
    python pipelines/train_spark.py --model gbt

    # Skip registry (still logs the run)
    python pipelines/train_spark.py --no-register
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import mlflow
import mlflow.spark
from mlflow.models.signature import infer_signature

from churnops.config import get_settings
from churnops.data.schema import TARGET_COL
from churnops.logging import setup_logging
from churnops.spark.session import get_spark, load_spark_config, read_split, stop_spark
from churnops.spark.train import metrics_table, train
from churnops.tracking.mlflow_utils import (
    log_data_params,
    log_model_params,
    log_split_metrics,
)
from churnops.tracking.registry import register_model, set_alias

setup_logging()
logger = logging.getLogger("churnops.pipeline.train_spark")

_DATA_YAML = _REPO_ROOT / "configs" / "data.yaml"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train Spark MLlib churn models and log to MLflow.")
    p.add_argument("--model", default=None, help="Single model key (default: all in spark.yaml)")
    p.add_argument("--tracking-uri", default=None, dest="tracking_uri", help="Override MLflow URI")
    p.add_argument("--no-register", action="store_true", help="Log the model but don't register it")
    return p.parse_args(argv)


def _setup_spark_experiment(cfg: dict, tracking_uri: str | None) -> str:
    """Point MLflow at the tracking server + the Spark-specific experiment."""
    mlflow_cfg = cfg["mlflow"]
    uri = tracking_uri or mlflow_cfg["tracking_uri"]
    mlflow.set_tracking_uri(uri)
    exp = mlflow_cfg["experiment_name"]
    if mlflow.get_experiment_by_name(exp) is None:
        mlflow.create_experiment(exp)
        logger.info("Created MLflow experiment: %s", exp)
    mlflow.set_experiment(exp)
    return uri


def _persist_native(model, model_key: str, cfg: dict) -> Path:
    """Save the fitted Spark PipelineModel to the artifacts dir (overwrite)."""
    base = _REPO_ROOT / cfg["artifacts"]["base_dir"] / model_key / cfg["artifacts"]["model_dir"]
    base.parent.mkdir(parents=True, exist_ok=True)
    model.write().overwrite().save(str(base))
    logger.info("Native Spark model saved → %s", base)
    return base


def run_one(model_key: str, cfg: dict, data_cfg: dict, tracking_uri: str | None, no_register: bool) -> str:
    settings = get_settings()
    mlflow_cfg = cfg["mlflow"]
    spark = get_spark()

    logger.info("=" * 60)
    logger.info("Training spark:%s", model_key)

    model, metrics = train(model_key=model_key, spark=spark)

    # Native Spark persistence (idempotent overwrite).
    _persist_native(model, model_key, cfg)

    # Signature from a small raw-input sample + its predictions.
    test_df = read_split(spark, "test")
    sample_in = test_df.drop(TARGET_COL).limit(5).toPandas()
    sample_out = model.transform(test_df.limit(5)).select("prediction").toPandas()
    signature = infer_signature(sample_in, sample_out)

    run_tags = {
        "model_key": model_key,
        "model_class": cfg["models"][model_key]["class"],
        "framework": "spark-mllib",
    }
    with mlflow.start_run(run_name=model_key, tags=run_tags) as run:
        run_id = run.info.run_id
        log_model_params(model_key, cfg["models"][model_key], settings.random_seed)
        log_data_params(data_cfg)
        log_split_metrics(metrics)

        # Text summary artifact.
        with tempfile.TemporaryDirectory() as tmp:
            summary = Path(tmp) / "summary.txt"
            summary.write_text(f"Model: spark:{model_key}\n{metrics_table(metrics)}\n")
            mlflow.log_artifact(str(summary), artifact_path="summary")

        mlflow.spark.log_model(model, artifact_path="model", signature=signature)

    if not no_register:
        version = register_model(
            run_id,
            model_artifact_path="model",
            model_name=mlflow_cfg["registered_model_name"],
            tags={"model_key": model_key, "framework": "spark-mllib"},
        )
        set_alias(version, mlflow_cfg["aliases"]["staging"], mlflow_cfg["registered_model_name"])
        server = tracking_uri or mlflow_cfg["tracking_uri"]
        print(
            f"  [spark:{model_key}]  "
            f"roc_auc(val={metrics['val']['roc_auc']:.4f} | test={metrics['test']['roc_auc']:.4f})  "
            f"v{version}  run={run_id[:8]}"
        )
        print(f"  URL: {server}/#/experiments  (experiment: {mlflow_cfg['experiment_name']})")
    else:
        print(
            f"  [spark:{model_key}]  "
            f"roc_auc(val={metrics['val']['roc_auc']:.4f} | test={metrics['test']['roc_auc']:.4f})  "
            f"run={run_id[:8]}"
        )

    print(metrics_table(metrics))
    return run_id


def run(argv=None) -> None:
    import yaml

    args = parse_args(argv)
    cfg = load_spark_config()
    with _DATA_YAML.open() as f:
        data_cfg = yaml.safe_load(f)

    tracking_uri = _setup_spark_experiment(cfg, args.tracking_uri)
    model_keys = [args.model] if args.model else list(cfg["models"].keys())

    print(f"\nTracking server : {tracking_uri}")
    print(f"Experiment      : {cfg['mlflow']['experiment_name']}")
    print(f"Registered model: {cfg['mlflow']['registered_model_name']}")
    print(f"Models to train : {model_keys}\n")

    try:
        run_ids = [run_one(k, cfg, data_cfg, args.tracking_uri, args.no_register) for k in model_keys]
    finally:
        stop_spark()

    print(f"\nAll done. {len(run_ids)} Spark run(s) logged.")
    print(f"Open MLflow UI: {tracking_uri}")


if __name__ == "__main__":
    run()
