# ruff: noqa: E402
"""promote_model.py — Compare registered versions and promote the best to production.

Usage
-----
    # Compare and promote (only promotes if best beats current production)
    python pipelines/promote_model.py

    # Force-promote even if not better
    python pipelines/promote_model.py --force

    # Dry run — show what would happen without changing anything
    python pipelines/promote_model.py --dry-run

    # Use a different tracking server
    python pipelines/promote_model.py --tracking-uri http://localhost:3000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from churnops.logging import setup_logging
from churnops.tracking.mlflow_utils import (
    get_promotion_metric,
    get_registered_model_name,
    load_mlflow_config,
    setup_tracking,
)
from churnops.tracking.registry import (
    get_alias,
    list_versions,
    load_production_model,
    promote_best,
)

setup_logging()


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Compare registered model versions and promote the best to production."
    )
    p.add_argument(
        "--tracking-uri",
        default=None,
        dest="tracking_uri",
        help="Override MLflow tracking URI",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Promote even if the best version does not beat the current production",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Print what would happen without changing aliases",
    )
    p.add_argument(
        "--metric",
        default=None,
        help="Metric name to compare (default: from configs/model.yaml)",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="After promotion, load the production model and score a sample row",
    )
    return p.parse_args(argv)


def run(argv=None) -> None:
    args = parse_args(argv)

    cfg = load_mlflow_config()
    setup_tracking(args.tracking_uri)

    model_name = get_registered_model_name()
    metric = args.metric or get_promotion_metric()
    prod_alias = cfg["aliases"]["production"]
    staging_alias = cfg["aliases"]["staging"]

    # ── List all versions ─────────────────────────────────────────────────────
    versions = list_versions(model_name, metric)
    if not versions:
        print(f"No versions found for '{model_name}'. Run train_mlflow.py first.")
        sys.exit(1)

    print(f"\nRegistered model : {model_name}")
    print(f"Promotion metric : {metric}")
    print(f"\n{'Version':<10} {'Metric':>10}  {'Aliases':<25}  {'Model key':<25}  Run ID")
    print("-" * 90)
    for v in versions:
        metric_str = f"{v.metric_value:.4f}" if v.metric_value is not None else "N/A"
        aliases_str = ", ".join(v.aliases) if v.aliases else "-"
        print(
            f"  v{v.version:<7} {metric_str:>10}  {aliases_str:<25}  "
            f"{v.model_key or '?':<25}  {v.run_id[:8]}"
        )

    current_prod = get_alias(prod_alias, model_name)
    print(f"\nCurrent '{prod_alias}' alias → v{current_prod or 'none'}")

    # ── Promote ───────────────────────────────────────────────────────────────
    new_ver, reason = promote_best(
        model_name=model_name,
        metric_name=metric,
        force=args.force,
        dry_run=args.dry_run,
    )

    if new_ver is None:
        print(f"\n⚠  No promotion: {reason}")
    else:
        tag = "[DRY RUN] " if args.dry_run else ""
        print(f"\n{tag}✓ Production alias → v{new_ver}")
        print(f"  Reason: {reason}")

        new_staging = get_alias(staging_alias, model_name)
        print(f"  Staging alias  → v{new_staging or 'none'}")

    # ── Optional: load production + score a sample row ────────────────────────
    if args.verify and not args.dry_run:
        print("\nVerifying: loading production model and scoring a sample row...")
        pipeline = load_production_model(model_name)
        sample = {
            "customerID": "VERIFY-001",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": 1,
            "Dependents": 0,
            "tenure": 24.0,
            "PhoneService": 1,
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "One year",
            "PaperlessBilling": 1,
            "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 58.5,
            "TotalCharges": 1404.0,
        }
        from churnops.models.inference import predict
        results = predict(sample, pipeline=pipeline)
        r = results[0]
        print(
            f"  Sample row → prediction={r.label}  "
            f"churn_probability={r.churn_probability:.4f}"
        )
        print("  Production model loaded and scoring successfully ✓")


if __name__ == "__main__":
    run()
