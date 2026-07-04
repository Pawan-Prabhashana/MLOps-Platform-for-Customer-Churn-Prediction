"""Tests for the monitoring layer: drift, performance, alerts, store, collector.

DB-optional and broker-free: every test uses an in-memory SQLite database
(via monitoring.db.get_engine) instead of live Postgres/Supabase, and the
collector test mocks out the reference-data/model loads instead of touching
Kafka or MLflow. No live services are required to run this file.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine

from churnops.monitoring import alerts, drift, performance, store
from churnops.monitoring.db import Alert, MonitoringRun, get_session_factory, init_db

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sqlite_engine():
    """A fresh, isolated in-memory SQLite engine with all monitoring tables created.

    Uses ``create_engine`` directly (not the lru_cached ``monitoring.db.get_engine``)
    so each test gets its own database with no cross-test cache sharing.
    """
    engine = create_engine("sqlite:///:memory:", future=True)
    init_db(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def session(sqlite_engine):
    factory = get_session_factory(sqlite_engine)
    s = factory()
    yield s
    s.close()


def _ts(minutes_ago: int) -> str:
    return (datetime.now(UTC) - timedelta(minutes=minutes_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _prediction_record(
    customer_id: str,
    proba: float,
    contract: str | None = "Month-to-month",
    actual_churn: str | None = None,
    minutes_ago: int = 0,
) -> dict[str, Any]:
    return {
        "customerID": customer_id,
        "churn_probability": proba,
        "prediction": "Yes" if proba >= 0.5 else "No",
        "event_ts": _ts(minutes_ago + 1),
        "processed_ts": _ts(minutes_ago),
        "actual_churn": actual_churn,
        "Contract": contract,
    }


# ── PSI / drift computation ───────────────────────────────────────────────────

class TestPSI:
    def test_identical_distributions_show_no_drift(self) -> None:
        rng = np.random.default_rng(42)
        reference = rng.normal(0.3, 0.1, size=2000)
        current = rng.normal(0.3, 0.1, size=2000)
        psi = drift.population_stability_index(reference, current)
        assert psi < 0.10
        assert drift.psi_band(psi) == "none"

    def test_shifted_distribution_shows_significant_drift(self) -> None:
        rng = np.random.default_rng(42)
        reference = rng.normal(0.2, 0.05, size=2000)
        current = rng.normal(0.8, 0.05, size=2000)
        psi = drift.population_stability_index(reference, current)
        assert psi > 0.25
        assert drift.psi_band(psi) == "significant"

    def test_categorical_psi_no_drift_when_proportions_match(self) -> None:
        reference = pd.Series(["A"] * 500 + ["B"] * 500)
        current = pd.Series(["A"] * 480 + ["B"] * 520)
        psi = drift.categorical_psi(reference, current)
        assert psi < 0.10

    def test_categorical_psi_drift_when_proportions_flip(self) -> None:
        reference = pd.Series(["A"] * 900 + ["B"] * 100)
        current = pd.Series(["A"] * 100 + ["B"] * 900)
        psi = drift.categorical_psi(reference, current)
        assert psi > 0.25

    def test_prediction_drift_report_skips_small_window(self) -> None:
        report = drift.prediction_drift_report(
            np.random.default_rng(0).random(1000),
            np.random.default_rng(1).random(5),
            min_sample_size=30,
        )
        assert report["skipped"] is True
        assert report["rows"] == []

    def test_prediction_drift_report_flags_significant_shift(self) -> None:
        rng = np.random.default_rng(7)
        reference = rng.beta(2, 8, size=1000)  # low churn probabilities
        current = rng.beta(8, 2, size=200)     # high churn probabilities
        report = drift.prediction_drift_report(reference, current, min_sample_size=30)
        assert report["skipped"] is False
        assert report["overall_drifted"] is True
        assert any(r["statistic_name"] == "psi" and r["is_drifted"] for r in report["rows"])


# ── Performance metrics ───────────────────────────────────────────────────────

class TestPerformanceMetrics:
    def test_perfect_predictions_score_1(self) -> None:
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_prob = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.3])
        y_pred = np.array([1, 1, 0, 0, 1, 0])
        metrics = performance.compute_performance_metrics(y_true, y_prob, y_pred)
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["roc_auc"] == 1.0

    def test_known_confusion_matrix(self) -> None:
        # 2 TP, 1 FP, 1 FN, 2 TN
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([1, 1, 0, 1, 0, 0])
        y_prob = np.array([0.9, 0.6, 0.4, 0.55, 0.2, 0.1])
        metrics = performance.compute_performance_metrics(y_true, y_prob, y_pred)
        assert metrics["accuracy"] == pytest.approx(4 / 6)
        assert metrics["precision"] == pytest.approx(2 / 3)  # 2 TP / (2 TP + 1 FP)
        assert metrics["recall"] == pytest.approx(2 / 3)      # 2 TP / (2 TP + 1 FN)

    def test_single_class_window_skips_roc_auc(self) -> None:
        y_true = np.array([1, 1, 1])
        y_prob = np.array([0.9, 0.8, 0.7])
        y_pred = np.array([1, 1, 1])
        metrics = performance.compute_performance_metrics(y_true, y_prob, y_pred)
        assert metrics["roc_auc"] is None

    def test_evaluate_degradation_flags_drop_below_tolerance(self) -> None:
        degraded, reason = performance.evaluate_degradation(0.60, baseline_roc_auc=0.80, tolerance=0.05)
        assert degraded is True
        degraded, reason = performance.evaluate_degradation(0.78, baseline_roc_auc=0.80, tolerance=0.05)
        assert degraded is False

    def test_performance_report_end_to_end(self) -> None:
        df = pd.DataFrame({
            "actual_churn": [1, 1, 1, 0, 0, 0] * 5,
            "churn_probability": [0.9, 0.6, 0.4, 0.55, 0.2, 0.1] * 5,
            "prediction": ["Yes", "Yes", "No", "Yes", "No", "No"] * 5,
            "processed_ts": pd.date_range("2026-01-01", periods=30, freq="min"),
        })
        report = performance.performance_report(
            df, baseline_roc_auc=0.80, degradation_tolerance=0.05, min_labeled_sample=10,
        )
        assert report["has_ground_truth"] is True
        assert report["sample_size"] == 30
        assert "roc_auc" in report["metrics"]
        assert len(report["rows"]) == len(report["metrics"])

    def test_performance_report_skips_when_no_ground_truth(self) -> None:
        df = pd.DataFrame({
            "actual_churn": [None, None, None],
            "churn_probability": [0.1, 0.2, 0.3],
            "prediction": ["No", "No", "No"],
        })
        report = performance.performance_report(df, baseline_roc_auc=0.80, min_labeled_sample=10)
        assert report["has_ground_truth"] is False
        assert report["rows"] == []


# ── Alerts ─────────────────────────────────────────────────────────────────────

class TestAlerts:
    def test_evaluate_alert_rules_fires_on_prediction_drift(self) -> None:
        prediction_drift = {
            "skipped": False,
            "psi_band": "significant",
            "rows": [
                {"feature_name": None, "drift_type": "prediction", "statistic_name": "psi",
                 "statistic_value": 0.40, "threshold": 0.25, "is_drifted": True},
            ],
        }
        fired = alerts.evaluate_alert_rules(
            prediction_drift=prediction_drift,
            data_drift={"skipped": True, "rows": []},
            performance={"has_ground_truth": False, "degraded": False},
            records_processed=500,
            config={"volume": {"min_records_per_window": 5, "max_records_per_window": 50000}, "severity": {}},
        )
        assert len(fired) == 1
        assert fired[0]["category"] == "drift"
        assert fired[0]["severity"] in ("warning", "critical")

    def test_severity_escalates_to_critical_past_multiplier(self) -> None:
        assert alerts.severity_for_breach(0.30, threshold=0.25, critical_multiplier=2.0) == "warning"
        assert alerts.severity_for_breach(0.60, threshold=0.25, critical_multiplier=2.0) == "critical"

    def test_severity_has_three_tiers(self) -> None:
        assert alerts.severity_for_breach(1.05, threshold=1.0, critical_multiplier=2.0, info_multiplier=1.2) == "info"
        assert alerts.severity_for_breach(1.5, threshold=1.0, critical_multiplier=2.0, info_multiplier=1.2) == "warning"
        assert alerts.severity_for_breach(2.5, threshold=1.0, critical_multiplier=2.0, info_multiplier=1.2) == "critical"

    def test_performance_degradation_fires_scaled_severity(self) -> None:
        # baseline=0.80, tolerance=0.05 -> floor=0.75. Severity scales by how
        # many "tolerance units" past that floor roc_auc has fallen.
        mild = {
            "has_ground_truth": True, "degraded": True, "reason": "mild dip",
            "metrics": {"roc_auc": 0.749}, "baseline_roc_auc": 0.80, "degradation_tolerance": 0.05,
        }
        severe = {
            "has_ground_truth": True, "degraded": True, "reason": "severe dip",
            "metrics": {"roc_auc": 0.40}, "baseline_roc_auc": 0.80, "degradation_tolerance": 0.05,
        }
        cfg = {"volume": {"min_records_per_window": 5, "max_records_per_window": 50000}, "severity": {}}

        mild_fired = alerts.evaluate_alert_rules(
            prediction_drift={"skipped": True, "rows": []}, data_drift={"skipped": True, "rows": []},
            performance=mild, records_processed=500, config=cfg,
        )
        severe_fired = alerts.evaluate_alert_rules(
            prediction_drift={"skipped": True, "rows": []}, data_drift={"skipped": True, "rows": []},
            performance=severe, records_processed=500, config=cfg,
        )
        assert mild_fired[0]["category"] == "performance"
        assert severe_fired[0]["category"] == "performance"
        assert mild_fired[0]["severity"] == "info"
        assert severe_fired[0]["severity"] == "critical"

    def test_volume_alert_fires_when_pipeline_stalls(self) -> None:
        fired = alerts.evaluate_alert_rules(
            prediction_drift={"skipped": True, "rows": []},
            data_drift={"skipped": True, "rows": []},
            performance={"has_ground_truth": False, "degraded": False},
            records_processed=0,
            config={"volume": {"min_records_per_window": 5, "max_records_per_window": 50000}, "severity": {}},
        )
        assert any(a["category"] == "volume" for a in fired)

    def test_fire_alerts_writes_alert_row(self, session) -> None:
        candidate = {
            "severity": "critical",
            "category": "drift",
            "message": "synthetic PSI breach for test",
            "metric_value": 0.42,
            "threshold": 0.25,
        }
        cfg = {"hooks": {"slack_webhook_env": "SLACK_WEBHOOK_URL", "smtp_enabled": False}}
        count = alerts.fire_alerts(session, [candidate], cfg, dry_run=False)
        session.commit()

        assert count == 1
        rows = session.query(Alert).all()
        assert len(rows) == 1
        assert rows[0].severity == "critical"
        assert rows[0].category == "drift"
        assert rows[0].acknowledged is False


# ── Timestamp normalization ───────────────────────────────────────────────────
#
# Regression coverage for a real bug: Postgres hands back timezone-*aware*
# datetimes for our DateTime(timezone=True) columns even though naive UTC
# values were inserted (psycopg2 attaches tzinfo on the way out). SQLite
# (used everywhere else in this file) does NOT reproduce that — it silently
# returns naive datetimes regardless of what went in — so the round-trip
# tests below can't catch this on their own; these two exercise the
# normalization helper directly instead.

class TestTimestampNormalization:
    def test_to_naive_utc_strips_tzinfo(self) -> None:
        aware = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        naive = datetime(2026, 1, 1, 12, 0, 0)
        assert store._to_naive_utc(aware) == naive
        assert store._to_naive_utc(aware).tzinfo is None
        assert store._to_naive_utc(naive) == naive  # already-naive passthrough
        assert store._to_naive_utc(None) is None

    def test_parse_ts_agrees_for_naive_and_aware_same_instant(self) -> None:
        # As if the DB round-trip attached UTC tzinfo to what we originally
        # sent in as a naive datetime (Postgres's actual behavior) — both
        # must normalize to the identical value for the dedup check to work.
        from_string = store._parse_ts("2026-01-01T12:00:00Z")
        from_aware_datetime = store._parse_ts(datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC))
        assert from_string == from_aware_datetime
        assert from_string.tzinfo is None


# ── store.py round-trips ───────────────────────────────────────────────────────

class TestStoreRoundtrip:
    def test_upsert_and_read_recent_predictions(self, session) -> None:
        records = [_prediction_record(f"CUST-{i:03d}", proba=0.1 * i, minutes_ago=i) for i in range(5)]
        inserted = store.upsert_predictions(session, records)
        session.commit()

        assert inserted == 5
        recent = store.get_recent_predictions(session, limit=10)
        assert len(recent) == 5
        assert recent[0]["customer_id"] == "CUST-000"  # most recently processed first

    def test_upsert_dedupes_on_customer_and_processed_ts(self, session) -> None:
        record = _prediction_record("CUST-DUPE", proba=0.5, minutes_ago=0)
        first = store.upsert_predictions(session, [record])
        second = store.upsert_predictions(session, [record])
        session.commit()

        assert first == 1
        assert second == 0
        assert store.get_prediction_count(session) == 1

    def test_upsert_dedupes_collision_within_same_batch(self, session) -> None:
        # Two events for the same customer scored in the same second (both
        # sharing one processed_ts) — a single Kafka window can contain this,
        # since processed_ts only has second resolution. Without in-batch
        # dedup this trips the DB's unique constraint on the very first
        # insert (neither row exists yet, so the "already in DB" check alone
        # wouldn't catch it).
        shared_ts = _ts(0)
        record_a = _prediction_record("CUST-COLLIDE", proba=0.3, minutes_ago=0)
        record_a["processed_ts"] = shared_ts
        record_b = _prediction_record("CUST-COLLIDE", proba=0.9, minutes_ago=0)
        record_b["processed_ts"] = shared_ts

        inserted = store.upsert_predictions(session, [record_a, record_b])
        session.commit()

        assert inserted == 1
        assert store.get_prediction_count(session) == 1

    def test_churn_rate_by_contract_aggregates_correctly(self, session) -> None:
        records = [
            _prediction_record("A1", proba=0.9, contract="Month-to-month", minutes_ago=1),
            _prediction_record("A2", proba=0.8, contract="Month-to-month", minutes_ago=2),
            _prediction_record("A3", proba=0.1, contract="Month-to-month", minutes_ago=3),
            _prediction_record("A4", proba=0.9, contract="Two year", minutes_ago=4),
        ]
        store.upsert_predictions(session, records)
        session.commit()

        rates = store.get_churn_rate_by_contract(session)
        assert rates["Month-to-month"]["total"] == 3
        assert rates["Month-to-month"]["churn_rate_pct"] == pytest.approx(2 / 3 * 100, abs=0.01)
        assert rates["Two year"]["total"] == 1
        assert rates["Two year"]["churn_rate_pct"] == 100.0

    def test_top_k_by_probability_orders_correctly(self, session) -> None:
        records = [_prediction_record(f"C{i}", proba=p, minutes_ago=i) for i, p in enumerate([0.1, 0.9, 0.5, 0.7, 0.3])]
        store.upsert_predictions(session, records)
        session.commit()

        top = store.get_top_k_by_probability(session, k=3)
        probs = [r["churn_probability"] for r in top]
        assert probs == sorted(probs, reverse=True)
        assert len(top) == 3
        assert probs[0] == pytest.approx(0.9)

    def test_backfill_contract_only_fills_missing(self, session) -> None:
        record = _prediction_record("CUST-BF", proba=0.4, contract=None, minutes_ago=0)
        store.upsert_predictions(session, [record])
        session.commit()

        updated = store.backfill_contract(session, {"CUST-BF": "One year"})
        session.commit()
        assert updated == 1

        recent = store.get_recent_predictions(session, limit=10)
        assert recent[0]["contract"] == "One year"


# ── collector.py end-to-end ────────────────────────────────────────────────────

class _FakeModel:
    """Deterministic P(churn) based on row order — enough to exercise drift math."""

    def predict_proba(self, X: Any) -> np.ndarray:
        n = len(X)
        rng = np.random.default_rng(123)
        proba = rng.beta(2, 5, size=n)
        return np.column_stack([1 - proba, proba])


class _FakeLoadedModel:
    def __init__(self) -> None:
        self.model = _FakeModel()
        self.source = "fake-for-tests"


def _fake_reference_df() -> pd.DataFrame:
    from churnops.data.schema import ALL_FEATURE_COLS

    rng = np.random.default_rng(0)
    n = 200
    data: dict[str, Any] = {}
    for col in ALL_FEATURE_COLS:
        if col in ("tenure", "MonthlyCharges", "TotalCharges"):
            data[col] = rng.uniform(0, 100, size=n)
        elif col in ("SeniorCitizen", "Partner", "Dependents", "PhoneService", "PaperlessBilling"):
            data[col] = rng.integers(0, 2, size=n)
        else:
            data[col] = rng.choice(["A", "B", "C"], size=n)
    return pd.DataFrame(data)


class TestCollectorCycle:
    def test_run_cycle_dry_run_does_not_persist(self, sqlite_engine, monkeypatch) -> None:
        from churnops.monitoring import collector

        monkeypatch.setattr(collector, "get_engine", lambda url=None: sqlite_engine)

        factory = get_session_factory(sqlite_engine)
        seed_session = factory()
        records = [_prediction_record(f"SEED-{i}", proba=0.1 * (i % 10), minutes_ago=i) for i in range(40)]
        store.upsert_predictions(seed_session, records)
        seed_session.commit()
        seed_session.close()

        monkeypatch.setattr(collector, "load_reference_data", lambda cfg: _fake_reference_df())
        monkeypatch.setattr(collector, "_resolve_model_version", lambda: "test-v1")
        monkeypatch.setattr(
            "churnops.streaming.model_loader.load_scoring_model",
            lambda source, **kwargs: _FakeLoadedModel(),
        )

        result = collector.run_cycle(source="db", window=100, dry_run=True)

        assert result["source"] == "db"
        assert result["status"] == "ok"
        assert result["records_processed"] == 40

        session_check = factory()
        assert session_check.query(MonitoringRun).count() == 0
        session_check.close()

    def test_run_cycle_end_to_end_with_shared_engine(self, sqlite_engine, monkeypatch) -> None:
        from churnops.monitoring import collector

        monkeypatch.setattr(collector, "get_engine", lambda url=None: sqlite_engine)

        factory = get_session_factory(sqlite_engine)
        seed_session = factory()
        records = [
            _prediction_record(f"SEED-{i}", proba=0.1 * (i % 10), actual_churn=("Yes" if i % 2 == 0 else "No"), minutes_ago=i)
            for i in range(40)
        ]
        store.upsert_predictions(seed_session, records)
        seed_session.commit()
        seed_session.close()

        monkeypatch.setattr(collector, "load_reference_data", lambda cfg: _fake_reference_df())
        monkeypatch.setattr(collector, "_resolve_model_version", lambda: "test-v1")
        monkeypatch.setattr(
            "churnops.streaming.model_loader.load_scoring_model",
            lambda source, **kwargs: _FakeLoadedModel(),
        )

        result = collector.run_cycle(source="db", window=100, dry_run=False)

        assert result["records_processed"] == 40
        assert result["status"] == "ok"
        assert result["performance"]["has_ground_truth"] is True
        # data drift always skipped for source="db" (needs a live raw-topic window)
        assert result["data_drift"]["skipped"] is True

        session_check = factory()
        runs = session_check.query(MonitoringRun).all()
        assert len(runs) == 1
        assert runs[0].records_processed == 40
        session_check.close()
