"""Thin wrappers over monitoring/store.py read helpers for the dashboard.

No SQL lives here — every function delegates to store.py and only does
light shaping (dict/list -> pandas DataFrame, simple aggregation over an
already-fetched window). No Streamlit import, so this module is testable
as plain Python and reusable outside the dashboard (e.g. a notebook).

Every ``fetch_*`` function takes ``database_url`` as its first argument
(never a live session/engine) so callers — in particular
``st.cache_data`` — can cache on a hashable, primitive key.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy import text

from churnops.monitoring import store
from churnops.monitoring.db import get_engine, get_session_factory

# The `predictions` table only carries a `contract` enrichment column today
# (see monitoring/db.py) — InternetService/PaymentMethod raw features are not
# persisted, so those secondary cuts can't be computed. Kept as a constant so
# the UI can render a graceful "not available" note instead of a blank chart.
UNAVAILABLE_SEGMENT_FEATURES: tuple[str, ...] = ("InternetService", "PaymentMethod")


class DashboardDataError(RuntimeError):
    """Raised when the monitoring store can't be reached at all.

    Distinct from "reachable but empty" (which callers should render as a
    friendly empty state, not an error) — this is specifically a
    connectivity problem the UI should show as a connection-error banner.
    """


def _connect(database_url: str | None):
    """Return a live engine, failing fast with a clear error if unreachable."""
    engine = get_engine(database_url)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        raise DashboardDataError(
            f"Cannot reach the monitoring store ({exc.__class__.__name__}): {exc}"
        ) from exc
    return engine


def _with_session(database_url: str | None, fn):
    engine = _connect(database_url)
    session = get_session_factory(engine)()
    try:
        return fn(session)
    finally:
        session.close()


# ── Fetchers ───────────────────────────────────────────────────────────────────

def fetch_recent_predictions(database_url: str | None, limit: int = 500) -> pd.DataFrame:
    rows = _with_session(database_url, lambda s: store.get_recent_predictions(s, limit=limit))
    return pd.DataFrame(rows)


def fetch_churn_rate_by_contract(database_url: str | None, limit: int = 5000) -> pd.DataFrame:
    result = _with_session(database_url, lambda s: store.get_churn_rate_by_contract(s, limit=limit))
    if not result:
        return pd.DataFrame(columns=["contract", "total", "churn_rate_pct"])
    return pd.DataFrame([{"contract": c, **v} for c, v in result.items()])


def fetch_top_k(database_url: str | None, k: int = 10, limit: int = 5000) -> pd.DataFrame:
    rows = _with_session(database_url, lambda s: store.get_top_k_by_probability(s, k=k, limit=limit))
    return pd.DataFrame(rows)


def fetch_prediction_volume(database_url: str | None, bucket: str = "hour", limit: int = 5000) -> pd.DataFrame:
    rows = _with_session(database_url, lambda s: store.get_prediction_volume(s, bucket=bucket, limit=limit))
    return pd.DataFrame(rows)


def fetch_latest_drift(database_url: str | None, limit: int = 50) -> pd.DataFrame:
    rows = _with_session(database_url, lambda s: store.get_latest_drift(s, limit=limit))
    return pd.DataFrame(rows)


def fetch_latest_performance(database_url: str | None, limit: int = 50) -> pd.DataFrame:
    rows = _with_session(database_url, lambda s: store.get_latest_performance(s, limit=limit))
    return pd.DataFrame(rows)


def fetch_active_alerts(
    database_url: str | None, limit: int = 50, unacknowledged_only: bool = True
) -> pd.DataFrame:
    rows = _with_session(
        database_url,
        lambda s: store.get_active_alerts(s, limit=limit, unacknowledged_only=unacknowledged_only),
    )
    return pd.DataFrame(rows)


def fetch_latest_monitoring_run(database_url: str | None) -> dict[str, Any] | None:
    return _with_session(database_url, lambda s: store.get_latest_monitoring_run(s))


def fetch_prediction_count(database_url: str | None) -> int:
    return _with_session(database_url, lambda s: store.get_prediction_count(s))


# ── Overview KPIs (with delta vs. the immediately-preceding window) ───────────

def _window_stats(df: pd.DataFrame, risk_threshold: float) -> dict[str, Any] | None:
    if df.empty:
        return None
    return {
        "total": len(df),
        "churn_rate_pct": float((df["prediction"] == "Yes").mean() * 100),
        "avg_probability": float(df["churn_probability"].mean()),
        "high_risk_count": int((df["churn_probability"] > risk_threshold).sum()),
    }


def fetch_overview_kpis(
    database_url: str | None, window: int = 500, risk_threshold: float = 0.70
) -> dict[str, Any]:
    """Current-window KPIs (one row per distinct customer) plus a delta
    against the immediately-preceding window of the same size.

    Reuses ``fetch_recent_predictions`` with a 2x limit and splits it in two
    (rows are already most-recent-first) — no new store.py query needed.
    Deduped to the latest event per customer_id *before* splitting, for the
    same reason as ``store._latest_per_customer``: these are customer-level
    KPIs ("customers scored"), and the demo pipeline frequently re-scores
    the same customer many times, which would otherwise inflate every KPI.
    """
    df = fetch_recent_predictions(database_url, limit=window * 2)
    empty = {
        "total": 0, "churn_rate_pct": None, "avg_probability": None, "high_risk_count": 0,
        "delta_total": None, "delta_churn_rate_pct": None,
        "delta_avg_probability": None, "delta_high_risk_count": None,
    }
    if df.empty:
        return empty

    if "customer_id" in df.columns:
        df = df.drop_duplicates(subset="customer_id", keep="first")

    current = _window_stats(df.iloc[:window], risk_threshold)
    previous = _window_stats(df.iloc[window : window * 2], risk_threshold)
    if current is None:
        return empty

    result = dict(current)
    for key in ("total", "churn_rate_pct", "avg_probability", "high_risk_count"):
        result[f"delta_{key}"] = (current[key] - previous[key]) if previous else None
    return result


def fetch_production_model_info(database_url: str | None = None) -> dict[str, Any]:
    """Best-effort production model version: MLflow registry first, then
    the most recent predictions.model_version, else "unknown". Never raises.
    """
    try:
        from churnops.tracking.mlflow_utils import (
            get_registered_model_name,
            load_mlflow_config,
            setup_tracking,
        )
        from churnops.tracking.registry import get_alias

        setup_tracking(None)
        name = get_registered_model_name()
        alias = load_mlflow_config()["aliases"]["production"]
        version = get_alias(alias, name)
        if version:
            return {"version": version, "model_name": name, "source": "mlflow"}
    except Exception:  # noqa: BLE001
        pass

    if database_url is not None:
        try:
            df = fetch_recent_predictions(database_url, limit=50)
            if not df.empty and "model_version" in df.columns and df["model_version"].notna().any():
                version = df["model_version"].dropna().iloc[0]
                return {"version": version, "model_name": None, "source": "predictions"}
        except DashboardDataError:
            pass

    return {"version": None, "model_name": None, "source": "unknown"}


# ── Risk banding ───────────────────────────────────────────────────────────────

def risk_band(probability: float | None, mid_threshold: float = 0.40, high_threshold: float = 0.70) -> str:
    """Classify a churn probability into "low" / "medium" / "high" / "unknown"."""
    if probability is None:
        return "unknown"
    if probability > high_threshold:
        return "high"
    if probability > mid_threshold:
        return "medium"
    return "low"


def risk_color(band: str, theme: dict[str, Any] | None = None) -> str:
    theme = theme or {}
    colors = {
        "high": theme.get("high_risk_color", "#D62728"),
        "medium": theme.get("mid_risk_color", "#E8A33D"),
        "low": theme.get("low_risk_color", "#2CA02C"),
        "unknown": "#888888",
    }
    return colors.get(band, "#888888")


# ── Export ────────────────────────────────────────────────────────────────────

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")
