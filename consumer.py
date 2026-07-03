#!/usr/bin/env python3
# ruff: noqa: E402
"""consumer.py — Kafka consumer that scores churn and publishes predictions.

Thin CLI wrapper around src/churnops/streaming/consumer_core.py. Consumes raw
customer events from telco.raw.customers, runs the trained churn model, and
publishes predictions to telco.churn.predictions. Records that can't be parsed,
validated, coerced, or scored are routed to telco.deadletter — one bad record
never kills the consumer.

Delivery semantics: AT-LEAST-ONCE. Offsets are committed only after a record is
fully processed (prediction published or dead-lettered).

Usage
-----
    # Streaming: subscribe and score continuously until Ctrl-C
    python consumer.py --mode streaming

    # Batch: consume up to N records (or end-of-topic), then print a summary
    python consumer.py --mode batch --max-records 1000

    # Force the local joblib model instead of the MLflow registry
    python consumer.py --mode streaming --model-source joblib

    # Score + print but publish nothing
    python consumer.py --mode batch --max-records 50 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from churnops.config import get_settings
from churnops.logging import setup_logging
from churnops.streaming.consumer_core import (
    render_summary_table,
    run_batch,
    run_streaming,
)
from churnops.streaming.kafka_clients import (
    consumer_from_config,
    load_kafka_config,
    producer_from_config,
)
from churnops.streaming.model_loader import load_scoring_model

setup_logging()
logger = logging.getLogger("churnops.consumer")


def _build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    c_cfg = load_kafka_config().get("consumer", {})

    parser = argparse.ArgumentParser(
        prog="consumer.py",
        description="Score churn on Kafka events and publish predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--mode", required=True, choices=["streaming", "batch"],
        help="streaming: poll continuously; batch: bounded window + summary.",
    )
    parser.add_argument(
        "--bootstrap-servers", default=settings.kafka_bootstrap_servers, metavar="HOSTS",
        help="Kafka bootstrap server(s).",
    )
    parser.add_argument(
        "--input-topic", default=settings.kafka_topic_raw_customers,
        help="Topic to consume raw events from.",
    )
    parser.add_argument(
        "--output-topic", default=settings.kafka_topic_predictions,
        help="Topic to publish predictions to.",
    )
    parser.add_argument(
        "--deadletter-topic", default=settings.kafka_topic_deadletter,
        help="Topic for records that fail processing.",
    )
    parser.add_argument(
        "--group-id", default=c_cfg.get("group_id", "churnops-consumer"),
        help="Consumer group id.",
    )
    parser.add_argument(
        "--model-source", default=c_cfg.get("model_source", "registry"),
        choices=["registry", "joblib"],
        help="Model source: registry (MLflow production alias) or local joblib.",
    )
    parser.add_argument(
        "--threshold", type=float, default=float(c_cfg.get("threshold", 0.5)),
        metavar="P", help="Churn decision threshold on P(churn).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Score and print but do NOT publish or dead-letter (and don't commit).",
    )
    parser.add_argument(
        "--max-records", type=int, default=int(c_cfg.get("max_records", 1000)),
        metavar="N", help="Batch mode: stop after N records (or end of topic).",
    )
    parser.add_argument(
        "--top-k", type=int, default=int(c_cfg.get("top_k", 10)),
        metavar="K", help="Batch summary: number of highest-risk customers to list.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    c_cfg = load_kafka_config().get("consumer", {})
    include_ground_truth = bool(c_cfg.get("include_ground_truth", True))
    poll_timeout_s = float(c_cfg.get("poll_timeout_s", 1.0))
    max_empty_polls = int(c_cfg.get("max_empty_polls", 5))
    progress_every = int(c_cfg.get("progress_every", 50))
    reports_dir = c_cfg.get("reports_dir", "artifacts/reports")

    logger.info(
        "Consumer starting: mode=%s  in=%s  out=%s  dlq=%s  group=%s  threshold=%.2f  dry_run=%s",
        args.mode, args.input_topic, args.output_topic, args.deadletter_topic,
        args.group_id, args.threshold, args.dry_run,
    )

    # ── Load the model ONCE at startup ────────────────────────────────────────
    loaded = load_scoring_model(args.model_source)
    logger.info("Model source in use: %s", loaded.source)

    # ── Build Kafka clients ───────────────────────────────────────────────────
    consumer = consumer_from_config(args.bootstrap_servers, args.group_id)
    producer = producer_from_config(args.bootstrap_servers)

    print(f"\nBootstrap : {args.bootstrap_servers}")
    print(f"Model     : {loaded.source}")
    print(f"Mode      : {args.mode}")
    print(f"Threshold : {args.threshold}\n")

    if args.mode == "streaming":
        summary = run_streaming(
            consumer, producer, loaded.model,
            input_topic=args.input_topic,
            output_topic=args.output_topic,
            deadletter_topic=args.deadletter_topic,
            threshold=args.threshold,
            include_ground_truth=include_ground_truth,
            poll_timeout_s=poll_timeout_s,
            dry_run=args.dry_run,
            progress_every=progress_every,
        )
    else:
        summary, batch_summary = run_batch(
            consumer, producer, loaded.model,
            input_topic=args.input_topic,
            output_topic=args.output_topic,
            deadletter_topic=args.deadletter_topic,
            threshold=args.threshold,
            include_ground_truth=include_ground_truth,
            max_records=args.max_records,
            poll_timeout_s=poll_timeout_s,
            max_empty_polls=max_empty_polls,
            top_k=args.top_k,
            reports_dir=reports_dir,
            write_report=not args.dry_run,
            dry_run=args.dry_run,
            progress_every=progress_every,
        )
        print("\n" + render_summary_table(batch_summary))

    print()
    print("=" * 60)
    print(f"  MODE          : {summary.mode}")
    print(f"  CONSUMED      : {summary.consumed}")
    print(f"  PREDICTED     : {summary.predicted}")
    print(f"  DEAD-LETTERED : {summary.dead_lettered}")
    print(f"  ELAPSED       : {summary.elapsed_s:.2f}s")
    print(f"  RATE          : {summary.effective_rate:.1f} msg/s")
    print("=" * 60)


if __name__ == "__main__":
    main()
