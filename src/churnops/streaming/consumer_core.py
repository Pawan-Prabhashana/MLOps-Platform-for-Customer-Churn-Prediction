"""Consumer scoring engine + streaming/batch run loops + batch summary.

Pipeline per record:
    raw bytes → JSON parse → validate required fields → light clean
              → score with the loaded pipeline → build prediction message

The clean step REUSES the data-layer ``clean_telco`` so the consumer applies
exactly the same coercions the model was trained on (Yes/No → 1/0, TotalCharges
blank → 0.0, numeric dtypes) rather than duplicating that logic.

Delivery semantics (documented): AT-LEAST-ONCE. The Kafka consumer is built
with ``enable.auto.commit=False``; we commit the offset only *after* a record
has been fully processed (prediction published or dead-lettered). A crash before
commit re-delivers the record on restart — no silent drops, at the cost of
possible duplicate reprocessing.
"""

from __future__ import annotations

import json
import logging
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from churnops.data.clean import clean_telco
from churnops.data.schema import ALL_FEATURE_COLS, ID_COL, TARGET_RAW
from churnops.streaming.deadletter import send_dead_letter
from churnops.streaming.serialization import now_utc

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent.parent.parent

# A valid event must carry the id plus every feature column the model expects.
# The ground-truth Churn field is optional (kept aside for monitoring).
REQUIRED_FIELDS: frozenset[str] = frozenset({ID_COL, *ALL_FEATURE_COLS})


# ── Errors ────────────────────────────────────────────────────────────────────

class ValidationError(ValueError):
    """Raised when an event is missing required fields."""


# ── Cleaning / scoring one event ──────────────────────────────────────────────

def clean_event(event: dict[str, Any]) -> tuple[pd.DataFrame, str | None]:
    """Validate + clean a single raw event into a model-ready feature frame.

    Returns:
        (features_df, actual_churn) where features_df has exactly
        ``ALL_FEATURE_COLS`` in canonical order and ``actual_churn`` is the
        ground-truth "Yes"/"No" (or None if the event didn't carry it).

    Raises:
        ValidationError: required fields missing.
        Exception:       any coercion failure from clean_telco (bad types etc.)
    """
    if not isinstance(event, dict):
        raise ValidationError(f"event is not a JSON object (got {type(event).__name__})")

    missing = REQUIRED_FIELDS - event.keys()
    if missing:
        raise ValidationError(f"missing required fields: {sorted(missing)}")

    actual_churn = event.get(TARGET_RAW)
    normalized_actual = actual_churn if actual_churn in ("Yes", "No") else None

    # Build a one-row raw frame. clean_telco needs a Churn column to encode the
    # target; inject a synthetic "No" when the event has no ground truth so the
    # shared cleaner runs unchanged — the resulting churn column is discarded.
    row: dict[str, Any] = {col: event.get(col) for col in ALL_FEATURE_COLS}
    row[ID_COL] = event[ID_COL]
    row[TARGET_RAW] = actual_churn if actual_churn is not None else "No"

    cleaned = clean_telco(pd.DataFrame([row]))
    features = cleaned[ALL_FEATURE_COLS].copy()
    return features, normalized_actual


def build_prediction_message(
    event: dict[str, Any],
    churn_probability: float,
    prediction: str,
    actual_churn: str | None,
    include_ground_truth: bool,
) -> dict[str, Any]:
    """Assemble the output message per the prediction contract."""
    msg: dict[str, Any] = {
        "customerID": event[ID_COL],
        "churn_probability": round(float(churn_probability), 4),
        "prediction": prediction,
        "event_ts": event.get("event_ts"),  # carried through unchanged
        "processed_ts": now_utc(),           # stamped at scoring time
    }
    if include_ground_truth and actual_churn is not None:
        msg["actual_churn"] = actual_churn
    return msg


@dataclass
class ProcessOutcome:
    ok: bool
    key: str | None
    prediction: dict[str, Any] | None = None
    event: dict[str, Any] | None = None
    error: str | None = None


def process_record(
    raw_value: bytes | str,
    model: Any,
    *,
    threshold: float,
    include_ground_truth: bool,
) -> ProcessOutcome:
    """Deserialize → validate → clean → score → build output for one record.

    Never raises — any failure is captured and returned as a not-ok outcome so
    the caller can dead-letter it and keep the consumer alive.
    """
    # 1. Parse JSON.
    try:
        text = raw_value.decode("utf-8") if isinstance(raw_value, bytes) else raw_value
        event = json.loads(text)
    except Exception as exc:  # noqa: BLE001
        return ProcessOutcome(ok=False, key=None, error=f"json_parse_error: {exc}")

    key = event.get(ID_COL) if isinstance(event, dict) else None

    # 2. Validate + clean + score.
    try:
        features, actual_churn = clean_event(event)
        proba = float(model.predict_proba(features)[:, 1][0])
        if not (0.0 <= proba <= 1.0):
            raise ValueError(f"probability out of range: {proba}")
        prediction = "Yes" if proba >= threshold else "No"
        msg = build_prediction_message(
            event, proba, prediction, actual_churn, include_ground_truth
        )
        return ProcessOutcome(ok=True, key=key, prediction=msg, event=event)
    except Exception as exc:  # noqa: BLE001
        return ProcessOutcome(
            ok=False,
            key=key,
            event=event if isinstance(event, dict) else None,
            error=f"{type(exc).__name__}: {exc}",
        )


# ── Run summary ───────────────────────────────────────────────────────────────

@dataclass
class ConsumerSummary:
    mode: str
    consumed: int = 0
    predicted: int = 0
    dead_lettered: int = 0
    elapsed_s: float = 0.0
    effective_rate: float = field(init=False)

    def __post_init__(self) -> None:
        self.effective_rate = self.consumed / self.elapsed_s if self.elapsed_s > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"[{self.mode}] consumed={self.consumed}  predicted={self.predicted}  "
            f"dead_lettered={self.dead_lettered}  elapsed={self.elapsed_s:.1f}s  "
            f"rate={self.effective_rate:.1f} msg/s"
        )


def _delivery_cb(err: Any, msg: Any) -> None:
    if err:
        logger.warning("Output delivery failed: %s", err)


def _publish_prediction(producer: Any, topic: str, msg: dict[str, Any]) -> None:
    key = str(msg["customerID"]).encode("utf-8")
    value = json.dumps(msg, ensure_ascii=False).encode("utf-8")
    producer.produce(topic, key=key, value=value, on_delivery=_delivery_cb)


# ── Streaming mode ────────────────────────────────────────────────────────────

def run_streaming(
    consumer: Any,
    producer: Any,
    model: Any,
    *,
    input_topic: str,
    output_topic: str,
    deadletter_topic: str,
    threshold: float,
    include_ground_truth: bool,
    poll_timeout_s: float = 1.0,
    dry_run: bool = False,
    progress_every: int = 50,
) -> ConsumerSummary:
    """Continuously poll, score, and publish predictions until SIGINT.

    Commits offsets after each record is processed (at-least-once).
    """
    consumer.subscribe([input_topic])
    counters = {"consumed": 0, "predicted": 0, "dead_lettered": 0}
    running = True

    def _sigint(sig: int, frame: Any) -> None:
        nonlocal running
        running = False
        logger.info("SIGINT received — stopping poll, flushing, committing…")

    old_handler = signal.signal(signal.SIGINT, _sigint)
    start = time.perf_counter()

    try:
        while running:
            msg = consumer.poll(poll_timeout_s)
            if msg is None:
                continue
            if msg.error() is not None:
                logger.warning("Consumer error: %s", msg.error())
                continue

            counters["consumed"] += 1
            _handle_one(
                msg, producer, model, output_topic, deadletter_topic,
                threshold, include_ground_truth, dry_run, counters,
            )

            if not dry_run:
                producer.poll(0)
                consumer.commit(message=msg, asynchronous=False)

            if counters["consumed"] % progress_every == 0:
                elapsed = time.perf_counter() - start
                rate = counters["consumed"] / elapsed if elapsed > 0 else 0
                logger.info(
                    "[streaming] consumed=%d  predicted=%d  dead_lettered=%d  rate=%.1f msg/s",
                    counters["consumed"], counters["predicted"],
                    counters["dead_lettered"], rate,
                )
    finally:
        signal.signal(signal.SIGINT, old_handler)
        if not dry_run:
            producer.flush()
        consumer.close()

    summary = ConsumerSummary(
        mode="streaming",
        consumed=counters["consumed"],
        predicted=counters["predicted"],
        dead_lettered=counters["dead_lettered"],
        elapsed_s=round(time.perf_counter() - start, 3),
    )
    logger.info("Run complete: %s", summary)
    return summary


# ── Batch mode ────────────────────────────────────────────────────────────────

def run_batch(
    consumer: Any,
    producer: Any,
    model: Any,
    *,
    input_topic: str,
    output_topic: str,
    deadletter_topic: str,
    threshold: float,
    include_ground_truth: bool,
    max_records: int = 1000,
    poll_timeout_s: float = 1.0,
    max_empty_polls: int = 5,
    top_k: int = 10,
    reports_dir: str | Path | None = None,
    write_report: bool = True,
    dry_run: bool = False,
    progress_every: int = 50,
) -> tuple[ConsumerSummary, dict[str, Any]]:
    """Consume a bounded window, score all records, then emit a batch summary.

    Termination: stops after ``max_records`` OR when ``max_empty_polls``
    consecutive polls each return no message (treated as end-of-topic) —
    whichever comes first. Tolerating several empty polls avoids stopping
    prematurely during the initial Kafka partition-assignment/rebalance window,
    when the first poll(s) legitimately return None before records arrive.

    Returns:
        (summary, batch_summary_dict). The batch summary is also printed and,
        when ``write_report`` is True, written to ``reports_dir`` as JSON + md.
    """
    consumer.subscribe([input_topic])
    counters = {"consumed": 0, "predicted": 0, "dead_lettered": 0}
    scored: list[dict[str, Any]] = []  # {event, prediction} pairs kept for summary
    start = time.perf_counter()
    empty_polls = 0

    try:
        while counters["consumed"] < max_records:
            msg = consumer.poll(poll_timeout_s)
            if msg is None:
                empty_polls += 1
                if empty_polls >= max_empty_polls:
                    logger.info(
                        "No new messages after %d empty polls — assuming end of topic.",
                        empty_polls,
                    )
                    break
                continue
            if msg.error() is not None:
                logger.warning("Consumer error: %s", msg.error())
                continue
            empty_polls = 0  # reset once real data flows

            counters["consumed"] += 1
            outcome = _handle_one(
                msg, producer, model, output_topic, deadletter_topic,
                threshold, include_ground_truth, dry_run, counters,
            )
            if outcome.ok and outcome.prediction is not None:
                scored.append({"event": outcome.event, "prediction": outcome.prediction})

            if not dry_run:
                producer.poll(0)
                consumer.commit(message=msg, asynchronous=False)

            if counters["consumed"] % progress_every == 0:
                logger.info(
                    "[batch] consumed=%d  predicted=%d  dead_lettered=%d",
                    counters["consumed"], counters["predicted"], counters["dead_lettered"],
                )
    finally:
        if not dry_run:
            producer.flush()
        consumer.close()

    summary = ConsumerSummary(
        mode="batch",
        consumed=counters["consumed"],
        predicted=counters["predicted"],
        dead_lettered=counters["dead_lettered"],
        elapsed_s=round(time.perf_counter() - start, 3),
    )
    logger.info("Run complete: %s", summary)

    batch_summary = summarize(scored, top_k=top_k, counters=counters)
    if write_report and scored:
        _write_reports(batch_summary, reports_dir)

    return summary, batch_summary


# ── Shared per-record handler ─────────────────────────────────────────────────

def _handle_one(
    msg: Any,
    producer: Any,
    model: Any,
    output_topic: str,
    deadletter_topic: str,
    threshold: float,
    include_ground_truth: bool,
    dry_run: bool,
    counters: dict[str, int],
) -> ProcessOutcome:
    """Score one Kafka message; publish prediction or dead-letter it."""
    raw = msg.value()
    outcome = process_record(
        raw, model, threshold=threshold, include_ground_truth=include_ground_truth
    )

    if outcome.ok and outcome.prediction is not None:
        counters["predicted"] += 1
        if dry_run:
            logger.info("[DRY-RUN] prediction=%s", json.dumps(outcome.prediction))
        else:
            _publish_prediction(producer, output_topic, outcome.prediction)
    else:
        counters["dead_lettered"] += 1
        if dry_run:
            logger.info("[DRY-RUN] would dead-letter key=%s error=%s", outcome.key, outcome.error)
        else:
            send_dead_letter(
                producer, deadletter_topic, raw, outcome.error or "unknown error", outcome.key
            )
    return outcome


# ── Batch summary ─────────────────────────────────────────────────────────────

def summarize(
    scored: list[dict[str, Any]],
    *,
    top_k: int = 10,
    counters: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Compute batch analytics: churn %, per-Contract churn %, top-K, means."""
    counters = counters or {}
    n = len(scored)

    if n == 0:
        return {
            "generated_ts": now_utc(),
            "counts": {
                "consumed": counters.get("consumed", 0),
                "predicted": counters.get("predicted", 0),
                "dead_lettered": counters.get("dead_lettered", 0),
            },
            "overall_churn_pct": 0.0,
            "mean_churn_probability": 0.0,
            "churn_pct_by_contract": {},
            "top_k_customers": [],
        }

    yes = sum(1 for r in scored if r["prediction"]["prediction"] == "Yes")
    mean_prob = sum(r["prediction"]["churn_probability"] for r in scored) / n

    # Per-Contract churn %.
    by_contract: dict[str, dict[str, int]] = {}
    for r in scored:
        contract = str((r["event"] or {}).get("Contract", "Unknown"))
        bucket = by_contract.setdefault(contract, {"total": 0, "yes": 0})
        bucket["total"] += 1
        if r["prediction"]["prediction"] == "Yes":
            bucket["yes"] += 1
    churn_pct_by_contract = {
        c: round(b["yes"] / b["total"] * 100, 2) for c, b in sorted(by_contract.items())
    }

    top = sorted(
        scored, key=lambda r: r["prediction"]["churn_probability"], reverse=True
    )[:top_k]
    top_k_customers = [
        {
            "customerID": r["prediction"]["customerID"],
            "churn_probability": r["prediction"]["churn_probability"],
            "prediction": r["prediction"]["prediction"],
        }
        for r in top
    ]

    return {
        "generated_ts": now_utc(),
        "counts": {
            "consumed": counters.get("consumed", n),
            "predicted": counters.get("predicted", n),
            "dead_lettered": counters.get("dead_lettered", 0),
        },
        "overall_churn_pct": round(yes / n * 100, 2),
        "mean_churn_probability": round(mean_prob, 4),
        "churn_pct_by_contract": churn_pct_by_contract,
        "top_k_customers": top_k_customers,
    }


def render_summary_table(summary: dict[str, Any]) -> str:
    """Return a clean plaintext rendering of the batch summary."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  BATCH PREDICTION SUMMARY")
    lines.append("=" * 60)
    c = summary["counts"]
    lines.append(f"  consumed      : {c['consumed']}")
    lines.append(f"  predicted     : {c['predicted']}")
    lines.append(f"  dead-lettered : {c['dead_lettered']}")
    lines.append(f"  overall churn : {summary['overall_churn_pct']}%")
    lines.append(f"  mean P(churn) : {summary['mean_churn_probability']}")
    lines.append("-" * 60)
    lines.append("  Churn % by Contract:")
    for contract, pct in summary["churn_pct_by_contract"].items():
        lines.append(f"    {contract:<20} {pct:>6}%")
    lines.append("-" * 60)
    lines.append(f"  Top {len(summary['top_k_customers'])} customers by churn probability:")
    lines.append(f"    {'customerID':<16} {'P(churn)':>10} {'pred':>6}")
    for row in summary["top_k_customers"]:
        lines.append(
            f"    {row['customerID']:<16} {row['churn_probability']:>10.4f} {row['prediction']:>6}"
        )
    lines.append("=" * 60)
    return "\n".join(lines)


def _summary_markdown(summary: dict[str, Any]) -> str:
    c = summary["counts"]
    md = [
        "# Batch Prediction Summary",
        "",
        f"_Generated: {summary['generated_ts']}_",
        "",
        "## Counts",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Consumed | {c['consumed']} |",
        f"| Predicted | {c['predicted']} |",
        f"| Dead-lettered | {c['dead_lettered']} |",
        f"| Overall churn % | {summary['overall_churn_pct']}% |",
        f"| Mean P(churn) | {summary['mean_churn_probability']} |",
        "",
        "## Churn % by Contract",
        "",
        "| Contract | Churn % |",
        "|----------|---------|",
    ]
    for contract, pct in summary["churn_pct_by_contract"].items():
        md.append(f"| {contract} | {pct}% |")
    md += [
        "",
        f"## Top {len(summary['top_k_customers'])} customers by churn probability",
        "",
        "| customerID | P(churn) | prediction |",
        "|------------|----------|------------|",
    ]
    for row in summary["top_k_customers"]:
        md.append(f"| {row['customerID']} | {row['churn_probability']} | {row['prediction']} |")
    md.append("")
    return "\n".join(md)


def _write_reports(summary: dict[str, Any], reports_dir: str | Path | None) -> tuple[Path, Path]:
    """Write the batch summary as JSON + markdown; return both paths."""
    d = Path(reports_dir) if reports_dir else (_REPO_ROOT / "artifacts" / "reports")
    if not d.is_absolute():
        d = _REPO_ROOT / d
    d.mkdir(parents=True, exist_ok=True)

    json_path = d / "batch_summary.json"
    md_path = d / "batch_summary.md"
    json_path.write_text(json.dumps(summary, indent=2))
    md_path.write_text(_summary_markdown(summary))
    logger.info("Batch summary written → %s , %s", json_path, md_path)
    return json_path, md_path
