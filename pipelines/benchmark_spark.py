# ruff: noqa: E402
"""benchmark_spark.py — Spark MLlib vs scikit-learn head-to-head.

Trains each comparable model pair on the SAME processed splits, records training
+ inference wall-clock time and key metrics, writes a markdown report + CSV + two
PNG charts to artifacts/reports/, and logs them all to MLflow.

Usage
-----
    python pipelines/benchmark_spark.py
    python pipelines/benchmark_spark.py --model gbt        # single pair
    python pipelines/benchmark_spark.py --no-mlflow        # skip MLflow logging
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import mlflow

from churnops.logging import setup_logging
from churnops.spark.benchmark import (
    COMMENTARY,
    plot_metrics,
    plot_timing,
    render_table,
    run_benchmark,
    write_csv,
    write_markdown,
)
from churnops.spark.session import load_spark_config, stop_spark

setup_logging()
logger = logging.getLogger("churnops.pipeline.benchmark_spark")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Benchmark Spark MLlib vs scikit-learn.")
    p.add_argument("--model", default=None, help="Single Spark model key (default: all pairs)")
    p.add_argument("--tracking-uri", default=None, dest="tracking_uri", help="Override MLflow URI")
    p.add_argument("--no-mlflow", action="store_true", help="Skip logging the report to MLflow")
    return p.parse_args(argv)


def run(argv=None) -> None:
    args = parse_args(argv)
    cfg = load_spark_config()

    spark_keys = [args.model] if args.model else None

    try:
        rows = run_benchmark(spark_keys)
    finally:
        stop_spark()

    csv_path = write_csv(rows)
    md_path = write_markdown(rows)
    metrics_png = plot_metrics(rows)
    timing_png = plot_timing(rows)

    print("\n" + render_table(rows))
    print("\n" + COMMENTARY.replace("> ", "").replace("**", ""))
    print(f"\nReport written:\n  {md_path}\n  {csv_path}\n  {metrics_png}\n  {timing_png}")

    if not args.no_mlflow:
        mlflow_cfg = cfg["mlflow"]
        uri = args.tracking_uri or mlflow_cfg["tracking_uri"]
        mlflow.set_tracking_uri(uri)
        exp = mlflow_cfg["experiment_name"]
        if mlflow.get_experiment_by_name(exp) is None:
            mlflow.create_experiment(exp)
        mlflow.set_experiment(exp)
        with mlflow.start_run(run_name="benchmark_spark_vs_sklearn", tags={"kind": "benchmark"}):
            for r in rows:
                prefix = f"{r.model_pair}_{r.framework}"
                mlflow.log_metrics({
                    f"{prefix}_train_time_s": r.train_time_s,
                    f"{prefix}_infer_time_s": r.infer_time_s,
                    f"{prefix}_roc_auc": r.roc_auc,
                    f"{prefix}_pr_auc": r.pr_auc,
                    f"{prefix}_f1": r.f1,
                })
            for p in (md_path, csv_path, metrics_png, timing_png):
                mlflow.log_artifact(str(p), artifact_path="benchmark")
        print(f"\nBenchmark logged to MLflow experiment: {exp}")


if __name__ == "__main__":
    run()
