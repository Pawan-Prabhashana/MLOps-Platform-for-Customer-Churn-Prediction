# ruff: noqa: E402
"""train_mlflow.py — Train configured models and log everything to MLflow.

Usage
-----
    # Train all models in configs/model.yaml
    python pipelines/train_mlflow.py

    # Train one specific model
    python pipelines/train_mlflow.py --model logistic_regression

    # Use a different tracking server
    python pipelines/train_mlflow.py --tracking-uri http://localhost:3000
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
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.models.signature import infer_signature

from churnops.config import get_settings
from churnops.data.schema import TARGET_COL
from churnops.logging import setup_logging
from churnops.models.persistence import save_pipeline
from churnops.models.train import metrics_table, train
from churnops.tracking.mlflow_utils import (
    log_data_params,
    log_model_params,
    log_split_metrics,
    start_run,
)
from churnops.tracking.plots import generate_all_plots
from churnops.tracking.registry import register_model, set_alias

setup_logging()
logger = logging.getLogger("churnops.pipeline.train_mlflow")

_DATA_YAML = _REPO_ROOT / "configs" / "data.yaml"
_MODEL_YAML = _REPO_ROOT / "configs" / "model.yaml"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train churn models and log runs to MLflow."
    )
    p.add_argument(
        "--model",
        default=None,
        help="Single model key to train (default: train all models in config)",
    )
    p.add_argument(
        "--tracking-uri",
        default=None,
        dest="tracking_uri",
        help="Override MLflow tracking URI",
    )
    p.add_argument(
        "--no-register",
        action="store_true",
        help="Log the model but do not register it in the Model Registry",
    )
    return p.parse_args(argv)


def _load_val_df(data_cfg: dict) -> pd.DataFrame:
    proc_dir = _REPO_ROOT / data_cfg["paths"]["processed_dir"]
    val_df = pd.read_parquet(proc_dir / data_cfg["paths"]["val_file"])
    return val_df.drop(columns=[TARGET_COL])


def run_one(
    model_key: str,
    data_cfg: dict,
    model_cfg: dict,
    mlflow_cfg: dict,
    tracking_uri: str | None,
    no_register: bool,
) -> str:
    """Train one model key, log to MLflow, optionally register. Returns run_id."""
    settings = get_settings()

    logger.info("=" * 60)
    logger.info("Training: %s", model_key)

    # ── Train (reuse existing logic) ──────────────────────────────────────────
    pipeline, metrics = train(model_key=model_key)

    # ── Load val split for signature inference ────────────────────────────────
    val_X = _load_val_df(data_cfg)
    sample_input = val_X.head(5)

    # ── Generate plots into a temp dir ────────────────────────────────────────
    proc_dir = _REPO_ROOT / data_cfg["paths"]["processed_dir"]
    test_df = pd.read_parquet(proc_dir / data_cfg["paths"]["test_file"])
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    with tempfile.TemporaryDirectory() as tmp:
        plot_dir = Path(tmp) / "plots"
        plot_paths = generate_all_plots(
            pipeline, y_test, y_pred, y_proba, plot_dir,
            split_name="test", model_key=model_key,
        )

        # ── Open MLflow run ───────────────────────────────────────────────────
        run_tags = {
            "model_key": model_key,
            "model_class": model_cfg["models"][model_key]["class"],
        }
        with start_run(run_name=model_key, tags=run_tags, tracking_uri=tracking_uri) as run:
            run_id = run.info.run_id

            # Params
            log_model_params(model_key, model_cfg["models"][model_key], settings.random_seed)
            log_data_params(data_cfg)

            # Metrics
            log_split_metrics(metrics)

            # Plots
            if mlflow_cfg.get("log_plots", True):
                for p in plot_paths:
                    mlflow.log_artifact(str(p), artifact_path="plots")

            # Sidecar JSON
            with tempfile.TemporaryDirectory() as sidecar_tmp:
                sidecar_dir = Path(sidecar_tmp)
                _, sidecar_path = save_pipeline(
                    pipeline, metrics,
                    pipeline_path=sidecar_dir / "pipeline.joblib",
                    sidecar_path=sidecar_dir / "pipeline_meta.json",
                )
                mlflow.log_artifact(str(sidecar_path), artifact_path="sidecar")

            # Metrics summary text
            summary_text = metrics_table(metrics)
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, dir=tmp
            ) as tf:
                tf.write(f"Model: {model_key}\n{summary_text}\n")
                summary_path = tf.name
            mlflow.log_artifact(summary_path, artifact_path="summary")

            # Log sklearn model with signature
            signature = infer_signature(sample_input, pipeline.predict(sample_input))
            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                signature=signature,
                input_example=sample_input.head(2),
            )

    # ── Register + tag with staging alias ────────────────────────────────────
    if not no_register:
        version = register_model(
            run_id,
            model_artifact_path="model",
            model_name=mlflow_cfg["registered_model_name"],
            tags={"model_key": model_key},
        )
        set_alias(version, mlflow_cfg["aliases"]["staging"], mlflow_cfg["registered_model_name"])

        tracking_server = tracking_uri or mlflow_cfg.get("tracking_uri", "http://localhost:3000")
        run_url = f"{tracking_server}/#/experiments/1/runs/{run_id}"
        print(
            f"  [{model_key}]  "
            f"roc_auc(val={metrics['val']['roc_auc']:.4f} | "
            f"test={metrics['test']['roc_auc']:.4f})  "
            f"v{version}  run={run_id[:8]}"
        )
        print(f"  URL: {run_url}")
    else:
        print(
            f"  [{model_key}]  "
            f"roc_auc(val={metrics['val']['roc_auc']:.4f} | "
            f"test={metrics['test']['roc_auc']:.4f})  run={run_id[:8]}"
        )

    return run_id


def run(argv=None) -> None:
    args = parse_args(argv)

    with _MODEL_YAML.open() as f:
        model_cfg = yaml.safe_load(f)
    with _DATA_YAML.open() as f:
        data_cfg = yaml.safe_load(f)
    mlflow_cfg = model_cfg["mlflow"]

    if args.tracking_uri:
        mlflow_cfg["tracking_uri"] = args.tracking_uri

    model_keys = (
        [args.model] if args.model
        else list(model_cfg["models"].keys())
    )

    print(f"\nTracking server : {args.tracking_uri or mlflow_cfg['tracking_uri']}")
    print(f"Experiment      : {mlflow_cfg['experiment_name']}")
    print(f"Models to train : {model_keys}\n")

    run_ids = []
    for key in model_keys:
        rid = run_one(
            key, data_cfg, model_cfg, mlflow_cfg,
            args.tracking_uri, args.no_register,
        )
        run_ids.append(rid)

    print(f"\nAll done. {len(run_ids)} run(s) logged.")
    print(f"Open MLflow UI: {args.tracking_uri or mlflow_cfg['tracking_uri']}")


if __name__ == "__main__":
    run()
