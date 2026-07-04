"""Write predictions/metrics/drift/alerts; read helpers for the dashboard.

Ingestion has two paths, kept decoupled from the scoring hot path:

1. Kafka: ``ingest_predictions_from_kafka`` consumes a bounded window of
   ``telco.churn.predictions`` under its own consumer group
   (``churnops-monitoring`` by default) and upserts rows into ``predictions``,
   deduped on (customer_id, processed_ts). This never touches the scoring
   consumer's group/offsets.
2. Direct call: ``log_prediction`` lets the FastAPI service or the consumer
   log a single prediction inline. It is fire-and-forget — any failure
   (DB down, bad row, etc.) is caught and logged, never raised, so it can
   never block or crash the scoring path that calls it.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from churnops.monitoring.db import (
    Alert,
    DriftMetric,
    MonitoringRun,
    PerformanceMetric,
    Prediction,
    get_engine,
    init_db,
    session_scope,
)

logger = logging.getLogger(__name__)


# ── Timestamp parsing ─────────────────────────────────────────────────────────

def _to_naive_utc(dt: datetime | None) -> datetime | None:
    """Normalize any datetime to naive UTC.

    Every timestamp in this system is UTC by convention (see
    ``serialization.now_utc``), but Postgres hands timezone-*aware* UTC
    datetimes back for our ``DateTime(timezone=True)`` columns even though we
    insert naive ones — psycopg2 attaches tzinfo on the way out. Without this
    normalization, a freshly-parsed naive datetime never compares equal to
    the same instant read back from the DB, silently breaking the
    (customer_id, processed_ts) dedup check on every call after the first.
    """
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone(UTC).replace(tzinfo=None)
    return dt


def _parse_ts(value: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp string (as emitted by serialization.now_utc).

    Returns None if `value` is missing or unparsable rather than raising —
    a malformed timestamp should never sink an otherwise-good prediction row.
    Always returns naive UTC (see ``_to_naive_utc``).
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return _to_naive_utc(value)
    try:
        return datetime.strptime(str(value), "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        try:
            return _to_naive_utc(datetime.fromisoformat(str(value).replace("Z", "+00:00")))
        except ValueError:
            logger.warning("Could not parse timestamp %r; storing as NULL.", value)
            return None


def _actual_churn_to_int(value: Any) -> int | None:
    """Normalize the ground-truth field ("Yes"/"No"/0/1/None) to 0/1/None."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        if value == "Yes":
            return 1
        if value == "No":
            return 0
    return None


# ── Generic bounded Kafka window consumption ──────────────────────────────────

def bounded_consume_topic(
    bootstrap_servers: str,
    topic: str,
    *,
    group_id: str,
    max_records: int = 500,
    poll_timeout_s: float = 1.0,
    max_empty_polls: int = 5,
    auto_offset_reset: str = "earliest",
    commit: bool = True,
) -> list[dict[str, Any]]:
    """Consume up to `max_records` JSON messages from `topic`, return parsed dicts.

    Stops after `max_records` OR after `max_empty_polls` consecutive empty
    polls (treated as end-of-window) — whichever comes first. When `commit`
    is False (dry-run), offsets are never committed so the run can be
    repeated without losing the window.
    """
    from churnops.streaming.kafka_clients import build_consumer

    consumer = build_consumer(
        bootstrap_servers, group_id, auto_offset_reset=auto_offset_reset
    )
    consumer.subscribe([topic])

    records: list[dict[str, Any]] = []
    empty_polls = 0
    try:
        while len(records) < max_records:
            msg = consumer.poll(poll_timeout_s)
            if msg is None:
                empty_polls += 1
                if empty_polls >= max_empty_polls:
                    break
                continue
            if msg.error() is not None:
                logger.warning("Consumer error on %s: %s", topic, msg.error())
                continue
            empty_polls = 0

            try:
                value = msg.value()
                text = value.decode("utf-8") if isinstance(value, bytes) else value
                records.append(json.loads(text))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                logger.warning("Skipping unparsable message on %s: %s", topic, exc)
                continue

            if commit:
                consumer.commit(message=msg, asynchronous=False)
    finally:
        consumer.close()

    logger.info(
        "bounded_consume_topic: topic=%s group=%s got=%d records (commit=%s)",
        topic, group_id, len(records), commit,
    )
    return records


# ── Predictions: write ─────────────────────────────────────────────────────────

def record_to_row_kwargs(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "customer_id": str(record.get("customerID") or record.get("customer_id")),
        "churn_probability": float(record["churn_probability"]),
        "prediction": str(record["prediction"]),
        "actual_churn": _actual_churn_to_int(record.get("actual_churn")),
        "event_ts": _parse_ts(record.get("event_ts")),
        "processed_ts": _parse_ts(record.get("processed_ts")),
        "model_source": record.get("model_source"),
        "model_version": record.get("model_version"),
        "contract": record.get("Contract") or record.get("contract"),
    }


def upsert_predictions(session: Session, records: list[dict[str, Any]]) -> int:
    """Insert `records` into `predictions`, deduped on (customer_id, processed_ts).

    Two passes: first collapse duplicate keys *within* the incoming batch
    itself (two events for the same customer can legitimately land in the
    same second — the producer's pacing is several events/sec but
    processed_ts has only second resolution — so a single Kafka window can
    contain the same (customer_id, processed_ts) pair twice), then skip any
    pair already present in the table. Both steps are needed: without the
    first, a same-batch collision trips the DB's unique constraint even
    though neither row exists yet. Skipping (rather than a dialect-specific
    ON CONFLICT clause) keeps this portable across SQLite (tests) and
    Postgres.
    """
    if not records:
        return 0

    rows = [record_to_row_kwargs(r) for r in records]

    deduped: dict[tuple[str, datetime | None], dict[str, Any]] = {}
    for row in rows:
        deduped[(row["customer_id"], row["processed_ts"])] = row
    n_batch_dupes = len(rows) - len(deduped)

    existing: set[tuple[str, datetime | None]] = set()
    if deduped:
        customer_ids = {k[0] for k in deduped}
        found = session.execute(
            select(Prediction.customer_id, Prediction.processed_ts).where(
                Prediction.customer_id.in_(customer_ids)
            )
        ).all()
        existing = {(c, _to_naive_utc(t)) for c, t in found}

    new_rows = [row for key, row in deduped.items() if key not in existing]
    for row in new_rows:
        session.add(Prediction(**row))
    if new_rows:
        session.flush()
    logger.info(
        "upsert_predictions: %d new, %d already-in-db, %d same-batch duplicates (of %d)",
        len(new_rows), len(deduped) - len(new_rows), n_batch_dupes, len(rows),
    )
    return len(new_rows)


def ingest_predictions_from_kafka(
    session: Session,
    *,
    bootstrap_servers: str,
    topic: str,
    group_id: str,
    max_records: int = 500,
    poll_timeout_s: float = 1.0,
    max_empty_polls: int = 5,
    dry_run: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    """Consume a bounded window from `topic` and upsert it into `predictions`.

    Returns (raw_records, inserted_count). In dry-run mode nothing is
    persisted (inserted_count is 0) and offsets are not committed.
    """
    records = bounded_consume_topic(
        bootstrap_servers,
        topic,
        group_id=group_id,
        max_records=max_records,
        poll_timeout_s=poll_timeout_s,
        max_empty_polls=max_empty_polls,
        commit=not dry_run,
    )
    inserted = 0
    if not dry_run and records:
        inserted = upsert_predictions(session, records)
    return records, inserted


def log_prediction(record: dict[str, Any], *, database_url: str | None = None) -> None:
    """Fire-and-forget: log a single prediction inline (API or consumer call site).

    Never raises — any failure (DB unreachable, malformed record) is caught
    and logged so the caller's scoring path is never blocked or broken by a
    monitoring-layer problem.
    """
    try:
        engine = get_engine(database_url)
        with session_scope(engine) as session:
            upsert_predictions(session, [record])
    except Exception:  # noqa: BLE001
        logger.warning(
            "log_prediction: failed to persist prediction for customerID=%s (non-fatal)",
            record.get("customerID"),
            exc_info=True,
        )


def backfill_contract(session: Session, contract_by_customer: dict[str, str]) -> int:
    """Backfill the `contract` column on existing rows, keyed by customer_id.

    Used by collector.py after fetching a raw-topic window for data drift —
    that window carries the `Contract` field, so we opportunistically fill it
    in on any matching rows that don't have it yet.
    """
    if not contract_by_customer:
        return 0
    updated = 0
    rows = session.execute(
        select(Prediction).where(
            Prediction.customer_id.in_(contract_by_customer.keys()),
            Prediction.contract.is_(None),
        )
    ).scalars().all()
    for row in rows:
        row.contract = contract_by_customer.get(row.customer_id)
        updated += 1
    if updated:
        session.flush()
    return updated


# ── Metrics / alerts / runs: write ────────────────────────────────────────────

def write_drift_metrics(session: Session, rows: list[dict[str, Any]]) -> int:
    for row in rows:
        session.add(DriftMetric(**row))
    if rows:
        session.flush()
    return len(rows)


def write_performance_metrics(session: Session, rows: list[dict[str, Any]]) -> int:
    for row in rows:
        session.add(PerformanceMetric(**row))
    if rows:
        session.flush()
    return len(rows)


def write_alert(session: Session, alert: dict[str, Any]) -> Alert:
    row = Alert(**alert)
    session.add(row)
    session.flush()
    return row


def write_monitoring_run(session: Session, run: dict[str, Any]) -> MonitoringRun:
    row = MonitoringRun(**run)
    session.add(row)
    session.flush()
    return row


# ── Read helpers (dashboard) ──────────────────────────────────────────────────

def get_recent_predictions(session: Session, limit: int = 200) -> list[dict[str, Any]]:
    rows = session.execute(
        select(Prediction).order_by(Prediction.processed_ts.desc().nullslast()).limit(limit)
    ).scalars().all()
    return [_prediction_to_dict(r) for r in rows]


def _latest_per_customer(rows: list[Prediction]) -> list[Prediction]:
    """Collapse a most-recent-first row list to one (the latest) row per customer_id.

    The demo pipeline frequently re-scores the same customer many times
    (batch-mode replays of the same CSV rows, or ordinary re-scoring over
    time) — for customer-population business metrics (churn rate by
    segment, the top-K action list) that means the same customer, not a
    fresh one. Event-level helpers (prediction volume, raw recent
    predictions) deliberately do NOT dedupe — those are about scoring
    *throughput*, where every event should count.
    """
    latest: dict[str, Prediction] = {}
    for r in rows:
        if r.customer_id not in latest:
            latest[r.customer_id] = r
    return list(latest.values())


def get_churn_rate_by_contract(session: Session, limit: int = 5000) -> dict[str, dict[str, float]]:
    """Churn rate (% predicted 'Yes') grouped by contract type, one vote per customer.

    Rows without a known contract (not yet backfilled by a data-drift run,
    see collector.py) are bucketed under "Unknown". See ``_latest_per_customer``
    for why this dedupes to the latest scoring event per customer.
    """
    rows = session.execute(
        select(Prediction)
        .order_by(Prediction.processed_ts.desc().nullslast())
        .limit(limit)
    ).scalars().all()

    buckets: dict[str, dict[str, int]] = {}
    for r in _latest_per_customer(rows):
        key = r.contract or "Unknown"
        b = buckets.setdefault(key, {"total": 0, "yes": 0})
        b["total"] += 1
        if r.prediction == "Yes":
            b["yes"] += 1

    return {
        c: {"total": b["total"], "churn_rate_pct": round(b["yes"] / b["total"] * 100, 2)}
        for c, b in sorted(buckets.items())
    }


def get_top_k_by_probability(session: Session, k: int = 10, limit: int = 5000) -> list[dict[str, Any]]:
    """Top-K distinct customers by their latest churn_probability.

    See ``_latest_per_customer`` — without this, a repeatedly-rescored
    customer would occupy multiple slots in what's meant to be a
    one-row-per-customer action list.
    """
    rows = session.execute(
        select(Prediction)
        .order_by(Prediction.processed_ts.desc().nullslast())
        .limit(limit)
    ).scalars().all()
    top = sorted(_latest_per_customer(rows), key=lambda r: r.churn_probability, reverse=True)[:k]
    return [_prediction_to_dict(r) for r in top]


def get_prediction_volume(session: Session, bucket: str = "hour", limit: int = 5000) -> list[dict[str, Any]]:
    """Prediction counts bucketed by time, most-recent bucket last.

    Bucketing is done in Python (not via a dialect-specific date_trunc) so
    this works identically against SQLite (tests), local Postgres, and
    Supabase.
    """
    fmt = "%Y-%m-%d %H:00" if bucket == "hour" else "%Y-%m-%d"
    rows = session.execute(
        select(Prediction.processed_ts)
        .where(Prediction.processed_ts.is_not(None))
        .order_by(Prediction.processed_ts.desc())
        .limit(limit)
    ).all()

    counts: dict[str, int] = {}
    for (ts,) in rows:
        key = ts.strftime(fmt)
        counts[key] = counts.get(key, 0) + 1

    return [{"bucket": k, "count": v} for k, v in sorted(counts.items())]


def get_latest_drift(session: Session, limit: int = 20) -> list[dict[str, Any]]:
    rows = session.execute(
        select(DriftMetric).order_by(DriftMetric.computed_at.desc()).limit(limit)
    ).scalars().all()
    return [_drift_to_dict(r) for r in rows]


def get_latest_performance(session: Session, limit: int = 20) -> list[dict[str, Any]]:
    rows = session.execute(
        select(PerformanceMetric).order_by(PerformanceMetric.computed_at.desc()).limit(limit)
    ).scalars().all()
    return [_performance_to_dict(r) for r in rows]


def get_prediction_count(session: Session) -> int:
    return session.execute(select(func.count()).select_from(Prediction)).scalar_one()


def get_active_alerts(
    session: Session, limit: int = 50, unacknowledged_only: bool = True
) -> list[dict[str, Any]]:
    """Most recent alerts, unacknowledged by default (the dashboard's action list)."""
    stmt = select(Alert).order_by(Alert.fired_at.desc()).limit(limit)
    if unacknowledged_only:
        stmt = select(Alert).where(Alert.acknowledged.is_(False)).order_by(Alert.fired_at.desc()).limit(limit)
    rows = session.execute(stmt).scalars().all()
    return [_alert_to_dict(r) for r in rows]


def get_latest_monitoring_run(session: Session) -> dict[str, Any] | None:
    """The most recent monitoring cycle's summary row, or None if none have run yet."""
    row = session.execute(
        select(MonitoringRun).order_by(MonitoringRun.run_ts.desc()).limit(1)
    ).scalar_one_or_none()
    return _monitoring_run_to_dict(row) if row else None


# ── Row → dict helpers ────────────────────────────────────────────────────────

def _prediction_to_dict(r: Prediction) -> dict[str, Any]:
    return {
        "id": r.id,
        "customer_id": r.customer_id,
        "churn_probability": r.churn_probability,
        "prediction": r.prediction,
        "actual_churn": r.actual_churn,
        "event_ts": r.event_ts.isoformat() if r.event_ts else None,
        "processed_ts": r.processed_ts.isoformat() if r.processed_ts else None,
        "model_source": r.model_source,
        "model_version": r.model_version,
        "contract": r.contract,
        "ingested_at": r.ingested_at.isoformat() if r.ingested_at else None,
    }


def _drift_to_dict(r: DriftMetric) -> dict[str, Any]:
    return {
        "id": r.id,
        "computed_at": r.computed_at.isoformat() if r.computed_at else None,
        "feature_name": r.feature_name,
        "drift_type": r.drift_type,
        "statistic_name": r.statistic_name,
        "statistic_value": r.statistic_value,
        "threshold": r.threshold,
        "is_drifted": r.is_drifted,
        "reference_window": r.reference_window,
        "current_window": r.current_window,
    }


def _performance_to_dict(r: PerformanceMetric) -> dict[str, Any]:
    return {
        "id": r.id,
        "computed_at": r.computed_at.isoformat() if r.computed_at else None,
        "metric_name": r.metric_name,
        "metric_value": r.metric_value,
        "sample_size": r.sample_size,
        "window_start": r.window_start.isoformat() if r.window_start else None,
        "window_end": r.window_end.isoformat() if r.window_end else None,
        "model_version": r.model_version,
    }


def _alert_to_dict(r: Alert) -> dict[str, Any]:
    return {
        "id": r.id,
        "fired_at": r.fired_at.isoformat() if r.fired_at else None,
        "severity": r.severity,
        "category": r.category,
        "message": r.message,
        "metric_value": r.metric_value,
        "threshold": r.threshold,
        "acknowledged": r.acknowledged,
    }


def _monitoring_run_to_dict(r: MonitoringRun) -> dict[str, Any]:
    return {
        "id": r.id,
        "run_ts": r.run_ts.isoformat() if r.run_ts else None,
        "records_processed": r.records_processed,
        "drift_detected": r.drift_detected,
        "performance_degraded": r.performance_degraded,
        "alerts_fired": r.alerts_fired,
        "status": r.status,
        "notes": r.notes,
    }


__all__ = [
    "bounded_consume_topic",
    "upsert_predictions",
    "ingest_predictions_from_kafka",
    "log_prediction",
    "backfill_contract",
    "write_drift_metrics",
    "write_performance_metrics",
    "write_alert",
    "write_monitoring_run",
    "get_recent_predictions",
    "get_churn_rate_by_contract",
    "get_top_k_by_probability",
    "get_prediction_volume",
    "get_latest_drift",
    "get_latest_performance",
    "get_prediction_count",
    "get_active_alerts",
    "get_latest_monitoring_run",
    "init_db",
]
