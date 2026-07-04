"""Tests for the dashboard data layer: charts.py builders + data_access.py wrappers.

Streamlit itself isn't unit-tested here (it's UI); most of what follows is
plain Python/pandas, so no Streamlit process needs to be running. data_access.py
tests use a temp-file SQLite store (unique per test via tmp_path) — no live
Postgres/Supabase needed. The one exception is TestAppSmoke, which uses
Streamlit's own headless `AppTest` runner to execute dashboard/app.py's actual
script and assert it raises nothing — this is what originally caught a real
bug (two identical `st.plotly_chart` calls across tabs collided on
Streamlit's auto-generated element ID) that none of the pure data-layer tests
below could have detected.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

import churnops.config as config_module
from churnops.dashboard import charts, data_access
from churnops.dashboard.data_access import DashboardDataError
from churnops.monitoring import store
from churnops.monitoring.db import get_engine, get_session_factory, init_db

_APP_PATH = Path(__file__).parent.parent / "dashboard" / "app.py"

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _ts(minutes_ago: int) -> str:
    return (datetime.now(UTC) - timedelta(minutes=minutes_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _prediction_record(customer_id: str, proba: float, contract: str = "Month-to-month", minutes_ago: int = 0) -> dict:
    return {
        "customerID": customer_id,
        "churn_probability": proba,
        "prediction": "Yes" if proba >= 0.5 else "No",
        "event_ts": _ts(minutes_ago + 1),
        "processed_ts": _ts(minutes_ago),
        "actual_churn": "Yes" if proba >= 0.5 else "No",
        "Contract": contract,
    }


@pytest.fixture
def empty_db_url(tmp_path) -> str:
    url = f"sqlite:///{tmp_path / 'empty.db'}"
    init_db(get_engine(url))
    return url


@pytest.fixture
def seeded_db_url(tmp_path) -> str:
    url = f"sqlite:///{tmp_path / 'seeded.db'}"
    engine = get_engine(url)
    init_db(engine)
    session = get_session_factory(engine)()

    records = [
        _prediction_record("C1", 0.92, "Month-to-month", minutes_ago=1),
        _prediction_record("C2", 0.81, "Month-to-month", minutes_ago=2),
        _prediction_record("C3", 0.15, "Two year", minutes_ago=3),
        _prediction_record("C4", 0.55, "One year", minutes_ago=4),
        _prediction_record("C5", 0.05, "Two year", minutes_ago=5),
    ]
    store.upsert_predictions(session, records)

    store.write_drift_metrics(session, [
        {
            "feature_name": None, "drift_type": "prediction", "statistic_name": "psi",
            "statistic_value": 0.03, "threshold": 0.25, "is_drifted": False,
            "reference_window": "train", "current_window": "last 5",
        },
        {
            "feature_name": "tenure", "drift_type": "data", "statistic_name": "psi",
            "statistic_value": 0.31, "threshold": 0.25, "is_drifted": True,
            "reference_window": "train", "current_window": "last 5",
        },
    ])
    store.write_performance_metrics(session, [
        {"metric_name": "roc_auc", "metric_value": 0.83, "sample_size": 5, "window_start": None, "window_end": None, "model_version": "1"},
        {"metric_name": "accuracy", "metric_value": 0.79, "sample_size": 5, "window_start": None, "window_end": None, "model_version": "1"},
    ])
    store.write_alert(session, {
        "severity": "critical", "category": "drift", "message": "tenure drifted",
        "metric_value": 0.31, "threshold": 0.25, "acknowledged": False,
    })
    store.write_monitoring_run(session, {
        "records_processed": 5, "drift_detected": True, "performance_degraded": False,
        "alerts_fired": 1, "status": "ok", "notes": "test run",
    })
    session.commit()
    session.close()
    return url


# ── charts.py ──────────────────────────────────────────────────────────────────

class TestChartsWithData:
    def test_churn_rate_bar_returns_figure(self) -> None:
        df = pd.DataFrame({"contract": ["Month-to-month", "Two year"], "churn_rate_pct": [45.0, 5.0]})
        fig = charts.churn_rate_bar(df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_probability_histogram_returns_figure(self) -> None:
        df = pd.DataFrame({"churn_probability": [0.1, 0.5, 0.9, 0.3, 0.7]})
        fig = charts.probability_histogram(df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_volume_timeseries_returns_figure(self) -> None:
        df = pd.DataFrame({"bucket": ["2026-01-01 00:00", "2026-01-01 01:00"], "count": [10, 20]})
        fig = charts.volume_timeseries(df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_drift_heatmap_returns_figure_with_feature_rows(self) -> None:
        df = pd.DataFrame({
            "feature_name": ["tenure", "MonthlyCharges", None],
            "statistic_name": ["psi", "psi", "psi"],
            "statistic_value": [0.31, 0.05, 0.02],
        })
        fig = charts.drift_heatmap(df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_risk_gauge_returns_figure(self) -> None:
        fig = charts.risk_gauge(0.73, mid_threshold=0.40, high_threshold=0.70)
        assert isinstance(fig, go.Figure)


class TestChartsEmptyInput:
    def test_churn_rate_bar_handles_empty_df(self) -> None:
        fig = charts.churn_rate_bar(pd.DataFrame())
        assert isinstance(fig, go.Figure)

    def test_probability_histogram_handles_empty_df(self) -> None:
        fig = charts.probability_histogram(pd.DataFrame())
        assert isinstance(fig, go.Figure)

    def test_volume_timeseries_handles_empty_df(self) -> None:
        fig = charts.volume_timeseries(pd.DataFrame())
        assert isinstance(fig, go.Figure)

    def test_drift_heatmap_handles_empty_df(self) -> None:
        fig = charts.drift_heatmap(pd.DataFrame())
        assert isinstance(fig, go.Figure)

    def test_drift_heatmap_handles_only_overall_rows(self) -> None:
        # Only the overall prediction-drift row (feature_name=None) — no
        # per-feature rows to plot in the heatmap.
        df = pd.DataFrame({
            "feature_name": [None], "statistic_name": ["psi"], "statistic_value": [0.03],
        })
        fig = charts.drift_heatmap(df)
        assert isinstance(fig, go.Figure)


# ── Risk banding ───────────────────────────────────────────────────────────────

class TestRiskBanding:
    def test_classifies_high_medium_low(self) -> None:
        assert data_access.risk_band(0.95, mid_threshold=0.40, high_threshold=0.70) == "high"
        assert data_access.risk_band(0.55, mid_threshold=0.40, high_threshold=0.70) == "medium"
        assert data_access.risk_band(0.10, mid_threshold=0.40, high_threshold=0.70) == "low"

    def test_boundary_values_are_exclusive_on_the_low_side(self) -> None:
        # > threshold, not >=
        assert data_access.risk_band(0.70, mid_threshold=0.40, high_threshold=0.70) == "medium"
        assert data_access.risk_band(0.40, mid_threshold=0.40, high_threshold=0.70) == "low"

    def test_none_probability_is_unknown(self) -> None:
        assert data_access.risk_band(None) == "unknown"

    def test_risk_color_returns_a_color_for_every_band(self) -> None:
        for band in ("high", "medium", "low", "unknown"):
            assert data_access.risk_color(band).startswith("#") or data_access.risk_color(band)


# ── data_access.py against a seeded SQLite store ──────────────────────────────

class TestDataAccessSeeded:
    def test_fetch_recent_predictions_shape(self, seeded_db_url) -> None:
        df = data_access.fetch_recent_predictions(seeded_db_url, limit=10)
        assert len(df) == 5
        assert "churn_probability" in df.columns

    def test_fetch_churn_rate_by_contract(self, seeded_db_url) -> None:
        df = data_access.fetch_churn_rate_by_contract(seeded_db_url)
        assert set(df["contract"]) == {"Month-to-month", "Two year", "One year"}
        mtm_rate = df.loc[df["contract"] == "Month-to-month", "churn_rate_pct"].iloc[0]
        assert mtm_rate == 100.0  # both Month-to-month rows predicted "Yes"

    def test_fetch_top_k_orders_by_probability(self, seeded_db_url) -> None:
        df = data_access.fetch_top_k(seeded_db_url, k=3)
        assert len(df) == 3
        assert list(df["churn_probability"]) == sorted(df["churn_probability"], reverse=True)
        assert df.iloc[0]["customer_id"] == "C1"

    def test_fetch_prediction_volume(self, seeded_db_url) -> None:
        df = data_access.fetch_prediction_volume(seeded_db_url, bucket="hour")
        assert not df.empty
        assert df["count"].sum() == 5

    def test_fetch_latest_drift_and_performance(self, seeded_db_url) -> None:
        drift_df = data_access.fetch_latest_drift(seeded_db_url)
        assert len(drift_df) == 2
        perf_df = data_access.fetch_latest_performance(seeded_db_url)
        assert len(perf_df) == 2

    def test_fetch_active_alerts(self, seeded_db_url) -> None:
        df = data_access.fetch_active_alerts(seeded_db_url)
        assert len(df) == 1
        assert df.iloc[0]["severity"] == "critical"

    def test_fetch_latest_monitoring_run(self, seeded_db_url) -> None:
        run = data_access.fetch_latest_monitoring_run(seeded_db_url)
        assert run["records_processed"] == 5
        assert run["drift_detected"] is True

    def test_fetch_overview_kpis(self, seeded_db_url) -> None:
        kpis = data_access.fetch_overview_kpis(seeded_db_url, window=5, risk_threshold=0.70)
        assert kpis["total"] == 5
        assert kpis["high_risk_count"] == 2  # C1 (0.92) and C2 (0.81)
        assert kpis["delta_total"] is None  # no previous window available (only 5 rows total)

    def test_fetch_overview_kpis_dedupes_repeated_customer(self, tmp_path) -> None:
        # A customer re-scored 3 times should count once toward "customers
        # scored" — otherwise repeated demo batch replays inflate every KPI.
        url = f"sqlite:///{tmp_path / 'dedupe_kpi.db'}"
        engine = get_engine(url)
        init_db(engine)
        session = get_session_factory(engine)()
        store.upsert_predictions(session, [
            _prediction_record("REPEAT", 0.9, minutes_ago=3),
            _prediction_record("REPEAT", 0.9, minutes_ago=2),
            _prediction_record("REPEAT", 0.9, minutes_ago=1),
            _prediction_record("OTHER", 0.1, minutes_ago=1),
        ])
        session.commit()
        session.close()

        kpis = data_access.fetch_overview_kpis(url, window=10, risk_threshold=0.70)
        assert kpis["total"] == 2

    def test_fetch_production_model_info_falls_back_when_mlflow_unavailable(self, seeded_db_url, monkeypatch) -> None:
        def _raise(*a, **k):
            raise RuntimeError("no mlflow server in this test")

        monkeypatch.setattr("churnops.tracking.mlflow_utils.setup_tracking", _raise)
        info = data_access.fetch_production_model_info(seeded_db_url)
        assert info["source"] in ("predictions", "unknown")


class TestDataAccessEmpty:
    def test_fetchers_return_empty_shapes_not_exceptions(self, empty_db_url) -> None:
        assert data_access.fetch_recent_predictions(empty_db_url).empty
        assert data_access.fetch_churn_rate_by_contract(empty_db_url).empty
        assert data_access.fetch_top_k(empty_db_url).empty
        assert data_access.fetch_prediction_volume(empty_db_url).empty
        assert data_access.fetch_latest_drift(empty_db_url).empty
        assert data_access.fetch_latest_performance(empty_db_url).empty
        assert data_access.fetch_active_alerts(empty_db_url).empty
        assert data_access.fetch_latest_monitoring_run(empty_db_url) is None
        assert data_access.fetch_prediction_count(empty_db_url) == 0

    def test_overview_kpis_on_empty_store(self, empty_db_url) -> None:
        kpis = data_access.fetch_overview_kpis(empty_db_url)
        assert kpis["total"] == 0
        assert kpis["churn_rate_pct"] is None
        assert kpis["high_risk_count"] == 0


class TestDataAccessConnectivity:
    def test_unreachable_store_raises_dashboard_data_error(self, tmp_path) -> None:
        bad_url = f"sqlite:////{tmp_path}/does/not/exist/db.sqlite3"
        with pytest.raises(DashboardDataError):
            data_access.fetch_prediction_count(bad_url)


# ── Full-app smoke test (headless Streamlit runner) ───────────────────────────

@pytest.fixture
def _app_database_url(tmp_path, monkeypatch):
    """Point dashboard/app.py's Settings.monitoring_database_url at a temp
    SQLite store for the duration of one test, then restore the process-wide
    get_settings() cache so later tests see the real environment again.
    """
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'app_smoke.db'}")
    config_module.get_settings.cache_clear()
    try:
        yield
    finally:
        config_module.get_settings.cache_clear()


class TestAppSmoke:
    def test_app_runs_without_exceptions_on_empty_store(self, _app_database_url, tmp_path) -> None:
        pytest.importorskip("streamlit")
        from streamlit.testing.v1 import AppTest

        init_db(get_engine(f"sqlite:///{tmp_path / 'app_smoke.db'}"))
        at = AppTest.from_file(str(_APP_PATH), default_timeout=60)
        at.run()
        assert not at.exception
        assert any("No data yet" in i.value for i in at.info)

    def test_app_runs_without_exceptions_on_populated_store(self, _app_database_url, tmp_path) -> None:
        pytest.importorskip("streamlit")
        from streamlit.testing.v1 import AppTest

        url = f"sqlite:///{tmp_path / 'app_smoke.db'}"
        engine = get_engine(url)
        init_db(engine)
        session = get_session_factory(engine)()
        store.upsert_predictions(session, [
            _prediction_record(f"S{i}", proba=0.1 * (i % 10), minutes_ago=i) for i in range(30)
        ])
        store.write_performance_metrics(session, [
            {"metric_name": "roc_auc", "metric_value": 0.83, "sample_size": 30,
             "window_start": None, "window_end": None, "model_version": "1"},
        ])
        store.write_drift_metrics(session, [
            {"feature_name": None, "drift_type": "prediction", "statistic_name": "psi",
             "statistic_value": 0.02, "threshold": 0.25, "is_drifted": False,
             "reference_window": "train", "current_window": "last 30"},
        ])
        session.commit()
        session.close()

        at = AppTest.from_file(str(_APP_PATH), default_timeout=60)
        at.run()
        assert not at.exception
        assert len(at.metric) >= 4  # at least the 4 Overview KPI cards
        assert len(at.tabs) == 5
