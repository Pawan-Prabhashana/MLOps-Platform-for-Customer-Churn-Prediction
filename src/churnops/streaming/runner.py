"""Core streaming and batch producer logic (broker-agnostic, testable).

Both ``run_streaming`` and ``run_batch`` accept a *producer* object that only
needs to expose ``.produce(topic, key, value, on_delivery)``, ``.poll(timeout)``,
and ``.flush()``.  This makes them trivially testable with a fake producer
(no live Kafka required).
"""

from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from churnops.streaming.events import (
    checkpoint_clear,
    checkpoint_load,
    checkpoint_save,
)
from churnops.streaming.serialization import extract_key, now_utc, row_to_bytes

logger = logging.getLogger(__name__)


# ── Minimal producer protocol (duck-typed for easy mocking) ──────────────────

@runtime_checkable
class ProducerLike(Protocol):
    def produce(
        self,
        topic: str,
        *,
        key: bytes,
        value: bytes,
        on_delivery: Any,
    ) -> None: ...

    def poll(self, timeout: float) -> int: ...

    def flush(self, timeout: float = 30.0) -> int: ...


# ── Run summary ───────────────────────────────────────────────────────────────

@dataclass
class RunSummary:
    mode: str
    sent: int = 0
    failed: int = 0
    elapsed_s: float = 0.0
    effective_rate: float = field(init=False)

    def __post_init__(self) -> None:
        self.effective_rate = self.sent / self.elapsed_s if self.elapsed_s > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"[{self.mode}] sent={self.sent}  failed={self.failed}  "
            f"elapsed={self.elapsed_s:.1f}s  rate={self.effective_rate:.1f} msg/s"
        )


# ── Delivery callback factory ─────────────────────────────────────────────────

def _make_delivery_cb(counters: dict[str, int]) -> Any:
    """Return a delivery callback that increments sent/failed counters."""
    def cb(err: Any, msg: Any) -> None:
        if err:
            counters["failed"] += 1
            logger.warning("Delivery failed: %s", err)
        else:
            counters["sent"] += 1
    return cb


# ── Streaming mode ────────────────────────────────────────────────────────────

def run_streaming(
    producer: Any,
    df: pd.DataFrame,
    topic: str,
    *,
    events_per_sec: float = 10.0,
    limit: int | None = None,
    dry_run: bool = False,
    progress_every: int = 100,
) -> RunSummary:
    """Continuously sample rows and publish them at the configured rate.

    Runs until ``limit`` is reached or SIGINT (Ctrl-C) is received.
    Each message gets a fresh ``event_ts`` = current UTC wall-clock time.

    Args:
        producer:       confluent-kafka Producer (or any :class:`ProducerLike`).
        df:             Full dataset DataFrame (rows sampled with replacement).
        topic:          Kafka topic name.
        events_per_sec: Target throughput (sleep-based token pacing).
        limit:          Stop after this many messages (None = run forever).
        dry_run:        Serialise and log but do not actually produce.
        progress_every: Log a progress line every N messages.

    Returns:
        A :class:`RunSummary` with final counts and timing.
    """
    interval = 1.0 / max(events_per_sec, 0.001)
    counters: dict[str, int] = {"sent": 0, "failed": 0}
    cb = _make_delivery_cb(counters)

    running = True

    def _sigint(sig: int, frame: Any) -> None:
        nonlocal running
        running = False
        logger.info("SIGINT received — flushing and exiting…")

    old_handler = signal.signal(signal.SIGINT, _sigint)
    start = time.perf_counter()

    try:
        n = 0  # local counter for pacing and progress
        while running and (limit is None or n < limit):
            t0 = time.perf_counter()

            row = df.sample(1, replace=True).iloc[0].to_dict()
            key = extract_key(row)
            value = row_to_bytes(row, event_ts=now_utc())

            if dry_run:
                logger.info("[DRY-RUN] key=%s value=%s…", key.decode(), value[:80].decode())
                counters["sent"] += 1
            else:
                producer.produce(topic, key=key, value=value, on_delivery=cb)
                producer.poll(0)

            n += 1
            if n % progress_every == 0:
                elapsed = time.perf_counter() - start
                rate = n / elapsed if elapsed > 0 else 0
                logger.info(
                    "[streaming] n=%d  sent=%d  failed=%d  rate=%.1f msg/s",
                    n, counters["sent"], counters["failed"], rate,
                )

            # Token pacing: sleep for whatever is left of the interval.
            elapsed_this = time.perf_counter() - t0
            sleep = interval - elapsed_this
            if sleep > 0:
                time.sleep(sleep)

    finally:
        signal.signal(signal.SIGINT, old_handler)
        if not dry_run:
            producer.flush()

    total_elapsed = time.perf_counter() - start
    summary = RunSummary(
        mode="streaming",
        sent=counters["sent"],
        failed=counters["failed"],
        elapsed_s=round(total_elapsed, 3),
    )
    logger.info("Run complete: %s", summary)
    return summary


# ── Batch mode ────────────────────────────────────────────────────────────────

def run_batch(
    producer: Any,
    df: pd.DataFrame,
    topic: str,
    *,
    batch_size: int = 500,
    limit: int | None = None,
    checkpoint_file: str,
    reset_checkpoint: bool = False,
    dry_run: bool = False,
    progress_every: int = 100,
) -> RunSummary:
    """Send the dataset in order, in chunks, with checkpoint/resume.

    Each row's ``event_ts`` is set to the UTC wall-clock time at send time.

    Checkpoint semantics:
        ``last_index`` = the *next* absolute row index to send.
        After each successful flush, the checkpoint advances by ``batch_size``.
        On restart (without ``--reset-checkpoint``), sending resumes from there.
        After completion the checkpoint file is deleted.

    ``--limit`` caps the number of rows sent **in this session** (not cumulative),
    so a partial first run followed by a resumed run with the same ``--limit``
    will send up to ``limit`` more rows starting from the checkpoint.

    Args:
        producer:          confluent-kafka Producer (or any :class:`ProducerLike`).
        df:                Full dataset DataFrame.
        topic:             Kafka topic name.
        batch_size:        Rows per flush chunk.
        limit:             Max rows to send this session.
        checkpoint_file:   Path to the JSON checkpoint file.
        reset_checkpoint:  If True, ignore any existing checkpoint and start over.
        dry_run:           Serialise and log but do not actually produce.
        progress_every:    Log a progress line every N messages.

    Returns:
        A :class:`RunSummary` with final counts and timing.
    """
    # ── Checkpoint init ───────────────────────────────────────────────────────
    if reset_checkpoint:
        checkpoint_clear(checkpoint_file)
        logger.info("Checkpoint reset — starting from row 0.")

    state = checkpoint_load(checkpoint_file)
    start_index = state["last_index"]
    prior_sent = state["total_sent"]

    if state.get("finished"):
        logger.info("Checkpoint shows dataset already finished. Use --reset-checkpoint to resend.")
        return RunSummary(mode="batch", sent=0, failed=0, elapsed_s=0.0)

    # ── Slice the rows for this session ──────────────────────────────────────
    remaining = df.iloc[start_index:]
    if limit is not None:
        remaining = remaining.iloc[:limit]

    total_rows = len(remaining)
    if total_rows == 0:
        logger.info("No rows to send (start_index=%d, dataset_size=%d).", start_index, len(df))
        return RunSummary(mode="batch", sent=0, failed=0, elapsed_s=0.0)

    logger.info(
        "Batch mode: resuming at row %d, sending up to %d rows in chunks of %d.",
        start_index, total_rows, batch_size,
    )

    counters: dict[str, int] = {"sent": 0, "failed": 0}
    cb = _make_delivery_cb(counters)
    start = time.perf_counter()
    session_sent = 0  # rows dispatched this session (pre-ack)

    for chunk_start in range(0, total_rows, batch_size):
        chunk = remaining.iloc[chunk_start: chunk_start + batch_size]

        for _, row in chunk.iterrows():
            key = extract_key(row.to_dict())
            value = row_to_bytes(row.to_dict(), event_ts=now_utc())

            if dry_run:
                logger.info("[DRY-RUN] key=%s value=%s…", key.decode(), value[:80].decode())
                counters["sent"] += 1
            else:
                producer.produce(topic, key=key, value=value, on_delivery=cb)
                producer.poll(0)

            session_sent += 1
            if session_sent % progress_every == 0:
                elapsed = time.perf_counter() - start
                rate = session_sent / elapsed if elapsed > 0 else 0
                logger.info(
                    "[batch] row=%d/%d  sent=%d  failed=%d  rate=%.1f msg/s",
                    start_index + chunk_start + len(chunk),
                    start_index + total_rows,
                    counters["sent"],
                    counters["failed"],
                    rate,
                )

        if not dry_run:
            producer.flush()

        # Advance checkpoint after each successful flush.
        next_abs_index = start_index + chunk_start + len(chunk)
        finished = next_abs_index >= len(df)
        if not dry_run:
            checkpoint_save(
                checkpoint_file,
                last_index=next_abs_index,
                total_sent=prior_sent + counters["sent"],
                finished=finished,
            )

    # Clean up checkpoint on normal completion.
    if not dry_run:
        is_finished = (start_index + total_rows) >= len(df)
        if is_finished or (limit is not None):
            checkpoint_clear(checkpoint_file)
            logger.info("Checkpoint cleared — batch complete.")

    total_elapsed = time.perf_counter() - start
    summary = RunSummary(
        mode="batch",
        sent=counters["sent"],
        failed=counters["failed"],
        elapsed_s=round(total_elapsed, 3),
    )
    logger.info("Run complete: %s", summary)
    return summary
