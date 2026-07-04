# ruff: noqa: E402
"""run_monitoring.py — Run one monitoring cycle: ingest, drift, performance, alerts.

Thin CLI wrapper around src/churnops/monitoring/collector.py. Pulls a bounded
window of scored predictions (from Kafka or the monitoring DB), computes
prediction drift (always) and data drift (when a raw-topic window is
reachable), computes performance metrics when ground truth is present,
persists everything to the monitoring store, evaluates alert rules, and
prints a console report.

Usage
-----
    # Consume a window from Kafka, compute + persist + alert (default)
    python pipelines/run_monitoring.py

    # Compute and print only — nothing is written, no offsets are committed
    python pipelines/run_monitoring.py --dry-run

    # Read the last 200 already-persisted predictions from the DB instead
    # (no Kafka needed; data drift is skipped since it needs a raw window)
    python pipelines/run_monitoring.py --source db --window 200

    # Point at a specific window size / Kafka cluster / DATABASE_URL
    python pipelines/run_monitoring.py --window 1000 --bootstrap-servers localhost:9092
    python pipelines/run_monitoring.py --database-url postgresql://...supabase.co:5432/postgres
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from churnops.config import get_settings
from churnops.logging import setup_logging
from churnops.monitoring.collector import load_monitoring_config, render_report, run_cycle

setup_logging()
logger = logging.getLogger("churnops.run_monitoring")


def _build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    cfg = load_monitoring_config()

    parser = argparse.ArgumentParser(
        prog="run_monitoring.py",
        description="Run one drift/performance monitoring cycle and persist the results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source", default="kafka", choices=["kafka", "db"],
        help="kafka: consume a fresh window from telco.churn.predictions. "
             "db: read the most recent window already in the monitoring store.",
    )
    parser.add_argument(
        "--window", type=int, default=cfg["ingestion"]["default_window"], metavar="N",
        help="Number of records to pull this cycle.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute and print only — no DB writes, no Kafka offset commits, no alerts persisted.",
    )
    parser.add_argument(
        "--bootstrap-servers", default=settings.kafka_bootstrap_servers, metavar="HOSTS",
        help="Kafka bootstrap server(s) (only used with --source kafka).",
    )
    parser.add_argument(
        "--database-url", default=None, metavar="URL",
        help="Override DATABASE_URL — e.g. a Supabase connection string. "
             "Defaults to settings.monitoring_database_url (local churnops Postgres).",
    )
    parser.add_argument(
        "--model-source", default="registry", choices=["registry", "joblib"],
        help="Model used to score the training reference for prediction drift.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logger.info(
        "Monitoring cycle starting: source=%s window=%d dry_run=%s",
        args.source, args.window, args.dry_run,
    )

    result = run_cycle(
        source=args.source,
        window=args.window,
        dry_run=args.dry_run,
        database_url=args.database_url,
        bootstrap_servers=args.bootstrap_servers,
        model_source=args.model_source,
    )

    print("\n" + render_report(result))

    if args.dry_run:
        print("\n[DRY-RUN] Nothing was written — no DB rows, no Kafka offset commits.\n")
    else:
        print(
            "\nInspect the tables with psql (or the Supabase SQL editor):\n"
            "  SELECT COUNT(*) FROM predictions;\n"
            "  SELECT COUNT(*) FROM drift_metrics;\n"
            "  SELECT COUNT(*) FROM performance_metrics;\n"
            "  SELECT COUNT(*) FROM monitoring_runs;\n"
            "  SELECT * FROM alerts ORDER BY fired_at DESC LIMIT 10;\n"
            "\n"
            "  # Against the local docker Postgres:\n"
            "  docker exec -it churnops-postgres psql -U churnops -d churnops\n"
            "\n"
            "To write to Supabase instead of the local DB, set DATABASE_URL in .env to your\n"
            "Supabase connection string (Project -> Settings -> Database -> Connection string)\n"
            "— same code, zero changes, it's plain Postgres either way.\n"
        )


if __name__ == "__main__":
    main()
