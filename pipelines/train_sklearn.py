# ruff: noqa: E402
"""train_sklearn.py — CLI: train, evaluate, and persist a sklearn Pipeline.

Usage
-----
    python pipelines/train_sklearn.py
    python pipelines/train_sklearn.py --model random_forest
    python pipelines/train_sklearn.py --model gradient_boosting --output artifacts/sklearn
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from churnops.logging import setup_logging
from churnops.models.persistence import save_pipeline
from churnops.models.train import metrics_table, train

setup_logging()
logger = logging.getLogger("churnops.pipeline.train_sklearn")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train a churn prediction Pipeline and save artifacts."
    )
    p.add_argument(
        "--model",
        default=None,
        help="Model key from configs/model.yaml (default: value of `default_model`)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Override output directory for artifacts (default: from config)",
    )
    return p.parse_args(argv)


def run(args=None) -> None:
    if args is None:
        args = parse_args()

    logger.info("Starting training run (model=%s)", args.model or "default")
    pipeline, metrics = train(model_key=args.model)

    # Resolve artifact paths
    pipe_path = sidecar_path = None
    if args.output:
        out = Path(args.output)
        pipe_path = out / "pipeline.joblib"
        sidecar_path = out / "pipeline_meta.json"

    saved_pipe, saved_side = save_pipeline(
        pipeline, metrics, pipeline_path=pipe_path, sidecar_path=sidecar_path
    )

    print(f"\nModel : {metrics['model_class']}")
    print(metrics_table(metrics))
    print("\nArtifacts")
    print(f"  Pipeline : {saved_pipe}")
    print(f"  Sidecar  : {saved_side}")


if __name__ == "__main__":
    run()
