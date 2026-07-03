"""Tests for the Kafka consumer: scoring, threshold, dead-letter, batch summary.

No live Kafka broker or MLflow server required — fake consumer/producer stubs
feed messages and record outputs, and a StubModel returns deterministic
probabilities. One test optionally uses the real joblib pipeline (fast, ~7 KB)
to prove unseen categoricals don't crash.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).parent.parent


# ── Fakes ─────────────────────────────────────────────────────────────────────

class FakeMessage:
    def __init__(self, value: bytes, key: bytes | None = None, err: Any = None) -> None:
        self._v = value
        self._k = key
        self._e = err

    def value(self) -> bytes:
        return self._v

    def key(self) -> bytes | None:
        return self._k

    def error(self) -> Any:
        return self._e

    def topic(self) -> str:
        return "telco.raw.customers"


class FakeConsumer:
    def __init__(self, raw_messages: list[bytes]) -> None:
        self._msgs = [FakeMessage(m) for m in raw_messages]
        self.subscribed: list[str] = []
        self.commits = 0
        self.closed = False

    def subscribe(self, topics: list[str]) -> None:
        self.subscribed = topics

    def poll(self, timeout: float = 1.0) -> FakeMessage | None:
        if self._msgs:
            return self._msgs.pop(0)
        return None

    def commit(self, message: Any = None, asynchronous: bool = True) -> None:
        self.commits += 1

    def close(self) -> None:
        self.closed = True


class FakeProducer:
    def __init__(self) -> None:
        self.produced: list[dict[str, Any]] = []

    def produce(self, topic: str, *, key: bytes, value: bytes, on_delivery: Any = None) -> None:
        self.produced.append({"topic": topic, "key": key, "value": value})
        if on_delivery is not None:
            on_delivery(None, None)

    def poll(self, timeout: float = 0) -> int:
        return 0

    def flush(self, timeout: float = 30.0) -> int:
        return 0

    def messages_on(self, topic: str) -> list[dict[str, Any]]:
        return [m for m in self.produced if m["topic"] == topic]


class StubModel:
    """Returns a fixed P(churn) for every row."""

    def __init__(self, proba: float) -> None:
        self.proba = proba

    def predict_proba(self, X: Any) -> np.ndarray:
        n = len(X)
        return np.array([[1.0 - self.proba, self.proba]] * n)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _valid_event(customer_id: str = "7590-VHVEG", churn: str | None = "No", **overrides: Any) -> dict:
    event = {
        "customerID": customer_id,
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
        "event_ts": "2025-10-03T04:00:00Z",
    }
    if churn is not None:
        event["Churn"] = churn
    event.update(overrides)
    return event


def _event_bytes(event: dict) -> bytes:
    return json.dumps(event).encode("utf-8")


TOPICS = {
    "input_topic": "telco.raw.customers",
    "output_topic": "telco.churn.predictions",
    "deadletter_topic": "telco.deadletter",
}


# ── Scoring / message contract ────────────────────────────────────────────────

class TestScoring:
    def test_valid_event_produces_well_formed_prediction(self) -> None:
        from churnops.streaming.consumer_core import process_record

        model = StubModel(0.82)
        outcome = process_record(
            _event_bytes(_valid_event()),
            model,
            threshold=0.5,
            include_ground_truth=True,
        )
        assert outcome.ok
        msg = outcome.prediction
        assert msg is not None
        assert msg["customerID"] == "7590-VHVEG"
        assert 0.0 <= msg["churn_probability"] <= 1.0
        assert msg["prediction"] in {"Yes", "No"}
        assert msg["event_ts"] == "2025-10-03T04:00:00Z"  # carried through
        assert "processed_ts" in msg and msg["processed_ts"].endswith("Z")

    def test_key_equals_customer_id(self) -> None:
        from churnops.streaming.consumer_core import run_batch

        consumer = FakeConsumer([_event_bytes(_valid_event("ABC-123"))])
        producer = FakeProducer()
        run_batch(
            consumer, producer, StubModel(0.9),
            threshold=0.5, include_ground_truth=True, write_report=False, **TOPICS,
        )
        preds = producer.messages_on("telco.churn.predictions")
        assert len(preds) == 1
        assert preds[0]["key"] == b"ABC-123"

    def test_ground_truth_included_when_present(self) -> None:
        from churnops.streaming.consumer_core import process_record

        outcome = process_record(
            _event_bytes(_valid_event(churn="Yes")),
            StubModel(0.3), threshold=0.5, include_ground_truth=True,
        )
        assert outcome.prediction["actual_churn"] == "Yes"

    def test_ground_truth_excluded_when_flag_off(self) -> None:
        from churnops.streaming.consumer_core import process_record

        outcome = process_record(
            _event_bytes(_valid_event(churn="Yes")),
            StubModel(0.3), threshold=0.5, include_ground_truth=False,
        )
        assert "actual_churn" not in outcome.prediction


class TestThreshold:
    def test_above_threshold_is_yes(self) -> None:
        from churnops.streaming.consumer_core import process_record

        outcome = process_record(
            _event_bytes(_valid_event()), StubModel(0.80),
            threshold=0.5, include_ground_truth=True,
        )
        assert outcome.prediction["prediction"] == "Yes"

    def test_below_threshold_is_no(self) -> None:
        from churnops.streaming.consumer_core import process_record

        outcome = process_record(
            _event_bytes(_valid_event()), StubModel(0.20),
            threshold=0.5, include_ground_truth=True,
        )
        assert outcome.prediction["prediction"] == "No"

    def test_custom_threshold_flips_decision(self) -> None:
        from churnops.streaming.consumer_core import process_record

        # proba 0.6: "Yes" at threshold 0.5, "No" at threshold 0.7
        low = process_record(
            _event_bytes(_valid_event()), StubModel(0.6),
            threshold=0.5, include_ground_truth=True,
        )
        high = process_record(
            _event_bytes(_valid_event()), StubModel(0.6),
            threshold=0.7, include_ground_truth=True,
        )
        assert low.prediction["prediction"] == "Yes"
        assert high.prediction["prediction"] == "No"


# ── Dead-letter handling ──────────────────────────────────────────────────────

class TestDeadLetter:
    def test_malformed_json_routed_to_deadletter(self) -> None:
        from churnops.streaming.consumer_core import run_batch

        consumer = FakeConsumer([b"this is { not json"])
        producer = FakeProducer()
        summary, _ = run_batch(
            consumer, producer, StubModel(0.5),
            threshold=0.5, include_ground_truth=True, write_report=False, **TOPICS,
        )
        # Nothing on the main output topic, one on dead-letter.
        assert producer.messages_on("telco.churn.predictions") == []
        dlq = producer.messages_on("telco.deadletter")
        assert len(dlq) == 1
        assert summary.dead_lettered == 1

    def test_deadletter_message_has_error_field(self) -> None:
        from churnops.streaming.consumer_core import run_batch

        consumer = FakeConsumer([b"{bad"])
        producer = FakeProducer()
        run_batch(
            consumer, producer, StubModel(0.5),
            threshold=0.5, include_ground_truth=True, write_report=False, **TOPICS,
        )
        dlq = producer.messages_on("telco.deadletter")
        payload = json.loads(dlq[0]["value"])
        assert "error" in payload
        assert "failed_ts" in payload
        assert "original" in payload

    def test_missing_required_field_deadlettered(self) -> None:
        from churnops.streaming.consumer_core import run_batch

        bad = _valid_event()
        del bad["tenure"]  # drop a required feature
        consumer = FakeConsumer([_event_bytes(bad)])
        producer = FakeProducer()
        run_batch(
            consumer, producer, StubModel(0.5),
            threshold=0.5, include_ground_truth=True, write_report=False, **TOPICS,
        )
        assert producer.messages_on("telco.churn.predictions") == []
        dlq = producer.messages_on("telco.deadletter")
        assert len(dlq) == 1
        payload = json.loads(dlq[0]["value"])
        assert "tenure" in payload["error"]

    def test_one_bad_record_does_not_stop_the_rest(self) -> None:
        from churnops.streaming.consumer_core import run_batch

        msgs = [
            _event_bytes(_valid_event("GOOD-1")),
            b"garbage",
            _event_bytes(_valid_event("GOOD-2")),
        ]
        consumer = FakeConsumer(msgs)
        producer = FakeProducer()
        summary, _ = run_batch(
            consumer, producer, StubModel(0.9),
            threshold=0.5, include_ground_truth=True, write_report=False, **TOPICS,
        )
        assert summary.consumed == 3
        assert summary.predicted == 2
        assert summary.dead_lettered == 1


# ── Batch summary ─────────────────────────────────────────────────────────────

class TestBatchSummary:
    def test_summarize_computes_churn_pct_and_topk(self) -> None:
        from churnops.streaming.consumer_core import summarize

        scored = [
            {"event": {"Contract": "Month-to-month"},
             "prediction": {"customerID": "A", "churn_probability": 0.9, "prediction": "Yes"}},
            {"event": {"Contract": "Month-to-month"},
             "prediction": {"customerID": "B", "churn_probability": 0.2, "prediction": "No"}},
            {"event": {"Contract": "Two year"},
             "prediction": {"customerID": "C", "churn_probability": 0.8, "prediction": "Yes"}},
        ]
        summary = summarize(scored, top_k=2)

        assert summary["overall_churn_pct"] == pytest.approx(66.67, abs=0.01)
        assert summary["mean_churn_probability"] == pytest.approx(0.6333, abs=0.001)
        assert summary["churn_pct_by_contract"]["Month-to-month"] == 50.0
        assert summary["churn_pct_by_contract"]["Two year"] == 100.0
        top_ids = [c["customerID"] for c in summary["top_k_customers"]]
        assert top_ids == ["A", "C"]  # sorted by prob desc, capped at 2

    def test_batch_writes_report_files(self, tmp_path: Path) -> None:
        from churnops.streaming.consumer_core import run_batch

        msgs = [_event_bytes(_valid_event(f"CUST-{i}")) for i in range(5)]
        consumer = FakeConsumer(msgs)
        producer = FakeProducer()
        run_batch(
            consumer, producer, StubModel(0.7),
            threshold=0.5, include_ground_truth=True,
            reports_dir=str(tmp_path), write_report=True, **TOPICS,
        )
        assert (tmp_path / "batch_summary.json").exists()
        assert (tmp_path / "batch_summary.md").exists()
        data = json.loads((tmp_path / "batch_summary.json").read_text())
        assert data["counts"]["predicted"] == 5


# ── Robustness: unseen categorical (real model) ───────────────────────────────

def _joblib_available() -> bool:
    return (_REPO_ROOT / "artifacts" / "sklearn" / "pipeline.joblib").exists()


@pytest.mark.skipif(not _joblib_available(), reason="joblib pipeline artifact not present")
class TestRealModelRobustness:
    def test_unseen_category_still_scores(self) -> None:
        from churnops.models.persistence import load_pipeline
        from churnops.streaming.consumer_core import process_record

        pipeline = load_pipeline()
        event = _valid_event(Contract="Quantum-entangled 999yr plan")
        outcome = process_record(
            _event_bytes(event), pipeline, threshold=0.5, include_ground_truth=True,
        )
        assert outcome.ok, outcome.error
        assert 0.0 <= outcome.prediction["churn_probability"] <= 1.0


# ── Dry-run ───────────────────────────────────────────────────────────────────

class TestDryRun:
    def test_dry_run_publishes_nothing(self) -> None:
        from churnops.streaming.consumer_core import run_batch

        msgs = [_event_bytes(_valid_event(f"CUST-{i}")) for i in range(4)]
        consumer = FakeConsumer(msgs)
        producer = FakeProducer()
        summary, _ = run_batch(
            consumer, producer, StubModel(0.9),
            threshold=0.5, include_ground_truth=True,
            write_report=False, dry_run=True, **TOPICS,
        )
        assert producer.produced == []       # nothing published or dead-lettered
        assert summary.predicted == 4        # but scoring still happened
