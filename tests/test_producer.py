"""Tests for the Kafka producer: serialization, batch/streaming modes, checkpoint.

No live Kafka broker required — a fake producer stub records all produce() calls
in memory so assertions are fast and deterministic.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).parent.parent


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """A tiny representative DataFrame that mirrors the real Telco CSV schema."""
    rows = [
        {
            "customerID": f"CUST-{i:04d}",
            "gender": "Female" if i % 2 == 0 else "Male",
            "SeniorCitizen": i % 2,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": i + 1,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 29.85 + i,
            "TotalCharges": 29.85 * (i + 1) if i != 3 else float("nan"),  # row 3 has blank TC
            "Churn": "No",
        }
        for i in range(20)
    ]
    return pd.DataFrame(rows)


@pytest.fixture()
def checkpoint_path(tmp_path: Path) -> Path:
    return tmp_path / "checkpoint.json"


# ── Fake producer ─────────────────────────────────────────────────────────────

class FakeProducer:
    """In-memory stub that mimics the confluent-kafka Producer interface."""

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.flush_count: int = 0

    def produce(
        self,
        topic: str,
        *,
        key: bytes,
        value: bytes,
        on_delivery: Any = None,
    ) -> None:
        self.messages.append({"topic": topic, "key": key, "value": value})
        if on_delivery is not None:
            on_delivery(None, self._fake_msg(topic, key, value))

    def poll(self, timeout: float = 0) -> int:
        return 0

    def flush(self, timeout: float = 30.0) -> int:
        self.flush_count += 1
        return 0

    @staticmethod
    def _fake_msg(topic: str, key: bytes, value: bytes) -> Any:
        """Minimal fake confluent Message object for the delivery callback."""
        class _Msg:
            def topic(self) -> str: return topic
            def key(self) -> bytes: return key
            def value(self) -> bytes: return value
        return _Msg()


# ── Serialization tests ───────────────────────────────────────────────────────

class TestSerialization:
    def test_key_is_customer_id_bytes(self, sample_df: pd.DataFrame) -> None:
        from churnops.streaming.serialization import extract_key

        row = sample_df.iloc[0].to_dict()
        key = extract_key(row)
        assert key == b"CUST-0000"
        assert isinstance(key, bytes)

    def test_value_is_valid_json(self, sample_df: pd.DataFrame) -> None:
        from churnops.streaming.serialization import row_to_bytes

        row = sample_df.iloc[0].to_dict()
        raw = row_to_bytes(row)
        msg = json.loads(raw)
        assert isinstance(msg, dict)

    def test_all_expected_fields_present(self, sample_df: pd.DataFrame) -> None:
        from churnops.streaming.serialization import EXPECTED_FIELDS, row_to_bytes

        row = sample_df.iloc[0].to_dict()
        msg = json.loads(row_to_bytes(row))
        assert EXPECTED_FIELDS.issubset(msg.keys())

    def test_event_ts_is_iso8601_utc(self, sample_df: pd.DataFrame) -> None:
        from churnops.streaming.serialization import row_to_bytes

        row = sample_df.iloc[0].to_dict()
        msg = json.loads(row_to_bytes(row))
        ts = msg["event_ts"]
        # Must end with Z and parse as ISO-8601.
        assert ts.endswith("Z"), f"event_ts should end with Z: {ts!r}"
        assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", ts), ts

    def test_explicit_event_ts_preserved(self, sample_df: pd.DataFrame) -> None:
        from churnops.streaming.serialization import row_to_bytes

        row = sample_df.iloc[0].to_dict()
        ts = "2025-10-03T04:00:00Z"
        msg = json.loads(row_to_bytes(row, event_ts=ts))
        assert msg["event_ts"] == ts

    def test_numeric_fields_are_numbers_not_strings(self, sample_df: pd.DataFrame) -> None:
        from churnops.streaming.serialization import row_to_bytes

        row = sample_df.iloc[0].to_dict()
        msg = json.loads(row_to_bytes(row))
        assert isinstance(msg["SeniorCitizen"], int)
        assert isinstance(msg["tenure"], int)
        assert isinstance(msg["MonthlyCharges"], float)
        assert isinstance(msg["TotalCharges"], float)

    def test_nan_total_charges_becomes_null(self, sample_df: pd.DataFrame) -> None:
        """Row 3 has NaN TotalCharges — must serialise as JSON null, not NaN."""
        from churnops.streaming.serialization import row_to_bytes

        row = sample_df.iloc[3].to_dict()  # has float('nan') for TotalCharges
        raw = row_to_bytes(row)
        # Must be valid JSON (would raise ValueError if NaN slipped through).
        msg = json.loads(raw)
        assert msg["TotalCharges"] is None

    def test_churn_field_preserved_raw(self, sample_df: pd.DataFrame) -> None:
        from churnops.streaming.serialization import row_to_bytes

        row = sample_df.iloc[0].to_dict()
        msg = json.loads(row_to_bytes(row))
        assert msg["Churn"] == "No"  # raw string, not encoded to 0/1


# ── Batch mode tests ──────────────────────────────────────────────────────────

class TestBatchMode:
    def test_sends_all_rows_in_order(
        self, sample_df: pd.DataFrame, checkpoint_path: Path
    ) -> None:
        from churnops.streaming.runner import run_batch

        producer = FakeProducer()
        run_batch(
            producer,
            sample_df,
            "test-topic",
            batch_size=5,
            limit=None,
            checkpoint_file=str(checkpoint_path),
        )
        assert producer.sent_count() == len(sample_df)

    def test_chunking_flushes_per_batch(
        self, sample_df: pd.DataFrame, checkpoint_path: Path
    ) -> None:
        from churnops.streaming.runner import run_batch

        producer = FakeProducer()
        # 20 rows, batch_size=7 → ceil(20/7) = 3 flushes
        run_batch(
            producer,
            sample_df,
            "test-topic",
            batch_size=7,
            limit=None,
            checkpoint_file=str(checkpoint_path),
        )
        assert producer.flush_count == 3

    def test_checkpoint_written_after_each_chunk(
        self, sample_df: pd.DataFrame, checkpoint_path: Path
    ) -> None:
        from churnops.streaming.runner import run_batch

        producer = FakeProducer()
        run_batch(
            producer,
            sample_df,
            "test-topic",
            batch_size=5,
            limit=10,  # send 10 rows
            checkpoint_file=str(checkpoint_path),
        )
        # After completion with limit, checkpoint is cleared.
        assert not checkpoint_path.exists()

    def test_resume_from_checkpoint_no_duplicates(
        self, sample_df: pd.DataFrame, checkpoint_path: Path
    ) -> None:
        from churnops.streaming.events import checkpoint_save
        from churnops.streaming.runner import run_batch

        # Simulate having sent rows 0-9 in a previous session.
        checkpoint_save(str(checkpoint_path), last_index=10, total_sent=10)

        producer = FakeProducer()
        run_batch(
            producer,
            sample_df,
            "test-topic",
            batch_size=5,
            limit=None,
            checkpoint_file=str(checkpoint_path),
        )

        # Only rows 10-19 should be sent (no duplicates of 0-9).
        assert producer.sent_count() == 10
        sent_ids = [
            json.loads(m["value"])["customerID"] for m in producer.messages
        ]
        # Should start from CUST-0010.
        assert sent_ids[0] == "CUST-0010"
        assert sent_ids[-1] == "CUST-0019"

    def test_reset_checkpoint_restarts_from_zero(
        self, sample_df: pd.DataFrame, checkpoint_path: Path
    ) -> None:
        from churnops.streaming.events import checkpoint_save
        from churnops.streaming.runner import run_batch

        # Previous run had sent 15 rows.
        checkpoint_save(str(checkpoint_path), last_index=15, total_sent=15)

        producer = FakeProducer()
        run_batch(
            producer,
            sample_df,
            "test-topic",
            batch_size=5,
            limit=5,
            checkpoint_file=str(checkpoint_path),
            reset_checkpoint=True,
        )

        # Should start from row 0 again.
        sent_ids = [json.loads(m["value"])["customerID"] for m in producer.messages]
        assert sent_ids[0] == "CUST-0000"

    def test_limit_caps_session_rows(
        self, sample_df: pd.DataFrame, checkpoint_path: Path
    ) -> None:
        from churnops.streaming.runner import run_batch

        producer = FakeProducer()
        run_batch(
            producer,
            sample_df,
            "test-topic",
            batch_size=5,
            limit=8,
            checkpoint_file=str(checkpoint_path),
        )
        assert producer.sent_count() == 8


# ── Streaming mode tests ──────────────────────────────────────────────────────

class TestStreamingMode:
    def test_limit_sends_exactly_n(self, sample_df: pd.DataFrame) -> None:
        from churnops.streaming.runner import run_streaming

        producer = FakeProducer()
        run_streaming(producer, sample_df, "test-topic", limit=7)
        assert producer.sent_count() == 7

    def test_each_message_has_fresh_timestamp(self, sample_df: pd.DataFrame) -> None:
        from churnops.streaming.runner import run_streaming

        producer = FakeProducer()
        run_streaming(producer, sample_df, "test-topic", limit=3)

        timestamps = [json.loads(m["value"])["event_ts"] for m in producer.messages]
        # All must be valid ISO-8601 UTC.
        iso_re = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
        for ts in timestamps:
            assert iso_re.match(ts), f"Bad timestamp: {ts!r}"

    def test_keys_are_customer_id_bytes(self, sample_df: pd.DataFrame) -> None:
        from churnops.streaming.runner import run_streaming

        producer = FakeProducer()
        run_streaming(producer, sample_df, "test-topic", limit=5)

        for m in producer.messages:
            payload = json.loads(m["value"])
            assert m["key"] == payload["customerID"].encode("utf-8")

    def test_returns_correct_summary(self, sample_df: pd.DataFrame) -> None:
        from churnops.streaming.runner import run_streaming

        producer = FakeProducer()
        summary = run_streaming(producer, sample_df, "test-topic", limit=10)
        assert summary.sent == 10
        assert summary.failed == 0
        assert summary.mode == "streaming"


# ── Dry-run tests ─────────────────────────────────────────────────────────────

class TestDryRun:
    def test_streaming_dry_run_sends_nothing(self, sample_df: pd.DataFrame) -> None:
        from churnops.streaming.runner import run_streaming

        producer = FakeProducer()
        run_streaming(producer, sample_df, "test-topic", limit=5, dry_run=True)
        # In dry-run the runner doesn't call producer.produce().
        assert len(producer.messages) == 0

    def test_batch_dry_run_sends_nothing(
        self, sample_df: pd.DataFrame, checkpoint_path: Path
    ) -> None:
        from churnops.streaming.runner import run_batch

        producer = FakeProducer()
        run_batch(
            producer,
            sample_df,
            "test-topic",
            batch_size=5,
            limit=10,
            checkpoint_file=str(checkpoint_path),
            dry_run=True,
        )
        assert len(producer.messages) == 0
        # Checkpoint should NOT be written during a dry run.
        assert not checkpoint_path.exists()


# ── Helpers added to FakeProducer ─────────────────────────────────────────────

def _sent_count(self: FakeProducer) -> int:
    return len(self.messages)


FakeProducer.sent_count = _sent_count  # type: ignore[attr-defined]
