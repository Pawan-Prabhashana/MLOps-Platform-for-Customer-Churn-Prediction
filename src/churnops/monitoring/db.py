"""SQLAlchemy engine/session + table models for the monitoring store.

Works against ANY Postgres — the local docker-compose `churnops` database or
a real Supabase project — through a single connection string
(``settings.monitoring_database_url`` / the ``DATABASE_URL`` env var). Tests
point this at an in-memory SQLite database instead; no live Postgres or
Supabase project is required to run them.

Deliberately uses the classic ``declarative_base()`` + ``Column()`` mapping
style rather than SQLAlchemy 2.0's ``DeclarativeBase``/``Mapped``/
``mapped_column`` — this module gets imported both from this project's own
environment (SQLAlchemy 2.x) *and* from inside the Airflow containers, which
hard-pin ``sqlalchemy<2.0`` (apache-airflow 2.9.2). The classic style runs
unchanged on both, so the monitoring DAG doesn't need its own virtualenv.
The same reasoning is why the Postgres driver is psycopg2 (universally
supported), not psycopg v3 (SQLAlchemy 2.0+ only — Airflow's bundled 1.4
has no dialect plugin for it at all).

Table overview
--------------
predictions          One row per scored record (log of everything the Kafka
                     consumer / API publishes). Optionally enriched with a
                     `contract` value backfilled from a raw-topic drift
                     window (see monitoring/collector.py) — this is the one
                     column added beyond the minimum needed for the
                     churn-rate-by-contract dashboard read helper; every
                     other predictions column is exactly ``{customer_id,
                     churn_probability, prediction, actual_churn, event_ts,
                     processed_ts, model_source, model_version,
                     ingested_at}``.
drift_metrics        One row per computed drift statistic (overall or
                     per-feature), data or prediction drift.
performance_metrics  One row per computed performance metric in a window.
monitoring_runs      One row per monitoring cycle (summary + status).
alerts               One row per fired alert.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(UTC)


Base = declarative_base()


# ── predictions ───────────────────────────────────────────────────────────────

class Prediction(Base):
    """Log of every record scored by the model (from Kafka or a direct call)."""

    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint("customer_id", "processed_ts", name="uq_predictions_customer_processed"),
        Index("ix_predictions_customer_id", "customer_id"),
        Index("ix_predictions_processed_ts", "processed_ts"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String(64), nullable=False)
    churn_probability = Column(Float, nullable=False)
    prediction = Column(String(8), nullable=False)
    actual_churn = Column(Integer, nullable=True)
    event_ts = Column(DateTime(timezone=True), nullable=True)
    processed_ts = Column(DateTime(timezone=True), nullable=True)
    model_source = Column(String(128), nullable=True)
    model_version = Column(String(32), nullable=True)
    # Additive enrichment column (not part of the message contract) used only
    # by the churn-rate-by-contract dashboard helper — see collector.py.
    contract = Column(String(32), nullable=True)
    ingested_at = Column(DateTime(timezone=True), default=_utcnow)


# ── drift_metrics ─────────────────────────────────────────────────────────────

class DriftMetric(Base):
    __tablename__ = "drift_metrics"
    __table_args__ = (Index("ix_drift_metrics_computed_at", "computed_at"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    computed_at = Column(DateTime(timezone=True), default=_utcnow)
    feature_name = Column(String(64), nullable=True)
    drift_type = Column(String(16), nullable=False)  # "data" | "prediction"
    statistic_name = Column(String(32), nullable=False)  # "psi" | "ks" | "chi_square"
    statistic_value = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    is_drifted = Column(Boolean, nullable=False, default=False)
    reference_window = Column(String(255), nullable=True)
    current_window = Column(String(255), nullable=True)


# ── performance_metrics ───────────────────────────────────────────────────────

class PerformanceMetric(Base):
    __tablename__ = "performance_metrics"
    __table_args__ = (Index("ix_performance_metrics_computed_at", "computed_at"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    computed_at = Column(DateTime(timezone=True), default=_utcnow)
    metric_name = Column(String(32), nullable=False)
    metric_value = Column(Float, nullable=False)
    sample_size = Column(Integer, nullable=False)
    window_start = Column(DateTime(timezone=True), nullable=True)
    window_end = Column(DateTime(timezone=True), nullable=True)
    model_version = Column(String(32), nullable=True)


# ── monitoring_runs ───────────────────────────────────────────────────────────

class MonitoringRun(Base):
    __tablename__ = "monitoring_runs"
    __table_args__ = (Index("ix_monitoring_runs_run_ts", "run_ts"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_ts = Column(DateTime(timezone=True), default=_utcnow)
    records_processed = Column(Integer, nullable=False, default=0)
    drift_detected = Column(Boolean, nullable=False, default=False)
    performance_degraded = Column(Boolean, nullable=False, default=False)
    alerts_fired = Column(Integer, nullable=False, default=0)
    status = Column(String(16), nullable=False, default="ok")
    notes = Column(Text, nullable=True)


# ── alerts ────────────────────────────────────────────────────────────────────

class Alert(Base):
    __tablename__ = "alerts"
    __table_args__ = (Index("ix_alerts_fired_at", "fired_at"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    fired_at = Column(DateTime(timezone=True), default=_utcnow)
    severity = Column(String(16), nullable=False)  # info|warning|critical
    category = Column(String(16), nullable=False)  # drift|performance|volume
    message = Column(Text, nullable=False)
    metric_value = Column(Float, nullable=True)
    threshold = Column(Float, nullable=True)
    acknowledged = Column(Boolean, nullable=False, default=False)


# ── Engine / session management ───────────────────────────────────────────────

def _normalize_url(database_url: str) -> str:
    """Rewrite a bare ``postgresql://`` URL to the explicit psycopg2 dialect.

    Supabase's dashboard hands you a plain ``postgresql://...`` connection
    string. SQLAlchemy defaults that to psycopg2 anyway when it's the only
    driver installed, but spelling it out avoids ambiguity and matches
    ``churnops.config.Settings.churnops_db_uri`` (already
    ``postgresql+psycopg2://``), so both forms resolve identically.
    """
    if database_url.startswith("postgresql://"):
        return "postgresql+psycopg2://" + database_url[len("postgresql://") :]
    return database_url


@functools.lru_cache(maxsize=8)
def get_engine(database_url: str | None = None) -> Engine:
    """Return a cached SQLAlchemy Engine for `database_url` (or the configured default)."""
    if database_url is None:
        from churnops.config import get_settings

        database_url = get_settings().monitoring_database_url

    url = _normalize_url(database_url)
    logger.debug("Creating monitoring DB engine for %s", url.split("@")[-1])
    return create_engine(url, pool_pre_ping=True, future=True)


def get_session_factory(engine: Engine) -> sessionmaker:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


@contextmanager
def session_scope(engine: Engine) -> Generator[Session, None, None]:
    """Context manager yielding a Session; commits on success, rolls back on error."""
    factory = get_session_factory(engine)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(engine: Engine) -> None:
    """Create all monitoring tables if they don't already exist (idempotent)."""
    Base.metadata.create_all(engine, checkfirst=True)
    logger.info("Monitoring tables ready (create_all, checkfirst=True).")


def clear_engine_cache() -> None:
    """Drop cached engines (useful in tests that swap DATABASE_URL)."""
    get_engine.cache_clear()
