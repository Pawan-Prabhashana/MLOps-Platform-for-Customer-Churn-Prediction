"""Orchestrates one monitoring cycle.

    ingest recent predictions -> compute prediction drift -> compute data
    drift (if a raw window is available) -> compute performance (if labels
    are present) -> persist everything -> evaluate alert rules -> write a
    monitoring_runs summary row -> return a result dict for the CLI to print.

Every step degrades gracefully: a missing training reference, an empty
window, an unreachable raw topic, or a labelless window all produce a
"skipped" result with a reason rather than raising — the only things that
raise are genuinely fatal setup problems (e.g. an unknown --source).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from churnops.monitoring import store
from churnops.monitoring.alerts import evaluate_alert_rules, fire_alerts
from churnops.monitoring.db import get_engine, get_session_factory, init_db
from churnops.monitoring.drift import data_drift_report, prediction_drift_report
from churnops.monitoring.performance import performance_report
from churnops.streaming.serialization import now_utc

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_MONITORING_YAML = _REPO_ROOT / "configs" / "monitoring.yaml"


# ── Config ────────────────────────────────────────────────────────────────────

def load_monitoring_config(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else _MONITORING_YAML
    with p.open() as f:
        return yaml.safe_load(f)


# ── Model / reference helpers ─────────────────────────────────────────────────

def _resolve_model_version() -> str | None:
    """Best-effort lookup of the production model's registry version. Never raises."""
    try:
        from churnops.tracking.mlflow_utils import (
            get_registered_model_name,
            load_mlflow_config,
            setup_tracking,
        )
        from churnops.tracking.registry import get_alias

        setup_tracking(None)  # ensure the tracking URI is set before querying the registry
        name = get_registered_model_name()
        alias = load_mlflow_config()["aliases"]["production"]
        return get_alias(alias, name)
    except Exception:  # noqa: BLE001
        return None


def _resolve_mlflow_baseline(model_version: str | None) -> float | None:
    """Fetch the production model version's logged test_roc_auc, if requested."""
    if model_version is None:
        return None
    try:
        from mlflow import MlflowClient

        from churnops.tracking.mlflow_utils import (
            get_run_metric,
            load_mlflow_config,
            setup_tracking,
        )

        setup_tracking(None)
        cfg = load_mlflow_config()
        client = MlflowClient()
        mv = client.get_model_version(cfg["registered_model_name"], model_version)
        return get_run_metric(mv.run_id, "test_roc_auc", client=client)
    except Exception:  # noqa: BLE001
        return None


def load_reference_data(cfg: dict[str, Any]) -> pd.DataFrame:
    """Load the training split — the reference for both prediction and data drift."""
    path = _REPO_ROOT / cfg["reference"]["train_parquet"]
    if not path.exists():
        raise FileNotFoundError(
            f"Training reference not found at {path}. Run pipelines/build_dataset.py first."
        )
    return pd.read_parquet(path)


def get_reference_prediction_distribution(reference_df: pd.DataFrame, model: Any) -> np.ndarray:
    """Score the training reference set with the model — the prediction-drift baseline."""
    from churnops.data.schema import ALL_FEATURE_COLS

    return model.predict_proba(reference_df[ALL_FEATURE_COLS])[:, 1]


def fetch_raw_drift_window(
    cfg: dict[str, Any],
    *,
    bootstrap_servers: str,
    raw_topic: str,
    max_records: int,
) -> pd.DataFrame | None:
    """Consume a bounded window of telco.raw.customers for feature-level data drift.

    Uses its own monitoring consumer group (never the scoring consumer's), so
    it can never disturb the real consumer's offsets. Returns None — never
    raises — when Kafka is unreachable or the window is empty; the caller
    then skips data drift for this cycle and logs why.
    """
    ingestion_cfg = cfg["ingestion"]
    try:
        records = store.bounded_consume_topic(
            bootstrap_servers,
            raw_topic,
            group_id=ingestion_cfg["group_id"] + "-raw",
            max_records=max_records,
            poll_timeout_s=ingestion_cfg["poll_timeout_s"],
            max_empty_polls=ingestion_cfg["max_empty_polls"],
            commit=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Data drift: could not consume raw window from %s (%s). Skipping data drift.",
            raw_topic, exc,
        )
        return None

    if not records:
        logger.info("Data drift: raw window from %s was empty. Skipping data drift.", raw_topic)
        return None

    from churnops.data.clean import clean_telco

    try:
        return clean_telco(pd.DataFrame(records))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Data drift: raw window failed cleaning (%s). Skipping data drift.", exc)
        return None


# ── Frame building ────────────────────────────────────────────────────────────

def _predictions_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Kafka-sourced prediction dicts -> the same shape get_recent_predictions returns."""
    if not records:
        return pd.DataFrame(
            columns=["customer_id", "churn_probability", "prediction", "actual_churn", "processed_ts"]
        )
    rows = [store.record_to_row_kwargs(r) for r in records]
    return pd.DataFrame(rows)


def _normalize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("processed_ts", "event_ts"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


# ── Main cycle ────────────────────────────────────────────────────────────────

def run_cycle(
    *,
    source: str = "kafka",
    window: int | None = None,
    dry_run: bool = False,
    database_url: str | None = None,
    bootstrap_servers: str | None = None,
    model_source: str = "registry",
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    from churnops.config import get_settings

    settings = get_settings()
    cfg = load_monitoring_config(config_path)

    if source not in ("kafka", "db"):
        raise ValueError(f"Unknown source '{source}'. Use 'kafka' or 'db'.")

    window = window or cfg["ingestion"]["default_window"]
    bootstrap = bootstrap_servers or settings.kafka_bootstrap_servers
    db_url = database_url or settings.monitoring_database_url

    engine = get_engine(db_url)
    if not dry_run:
        init_db(engine)
    session_factory = get_session_factory(engine)
    session = session_factory()

    result: dict[str, Any] = {
        "run_ts": now_utc(),
        "source": source,
        "window": window,
        "dry_run": dry_run,
        "database_url_host": db_url.split("@")[-1] if "@" in db_url else db_url,
    }

    try:
        # ── 1. Ingest ─────────────────────────────────────────────────────────
        inserted = 0
        if source == "kafka":
            records, inserted = store.ingest_predictions_from_kafka(
                session,
                bootstrap_servers=bootstrap,
                topic=settings.kafka_topic_predictions,
                group_id=cfg["ingestion"]["group_id"],
                max_records=window,
                poll_timeout_s=cfg["ingestion"]["poll_timeout_s"],
                max_empty_polls=cfg["ingestion"]["max_empty_polls"],
                dry_run=dry_run,
            )
            current_df = _normalize_timestamps(_predictions_to_frame(records))
        else:
            rows = store.get_recent_predictions(session, limit=window)
            current_df = _normalize_timestamps(pd.DataFrame(rows))

        records_processed = len(current_df)
        result["records_processed"] = records_processed
        result["records_inserted"] = inserted

        # ── 2. Reference data + model (best-effort; needed for both drifts) ────
        reference_df: pd.DataFrame | None = None
        model = None
        model_version = _resolve_model_version()
        try:
            reference_df = load_reference_data(cfg)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load training reference: %s", exc)

        # ── 3. Prediction drift ─────────────────────────────────────────────────
        pred_cfg = cfg["drift"]["prediction"]
        if records_processed == 0:
            prediction_drift = {
                "skipped": True, "reason": "no records in window", "rows": [], "overall_drifted": False,
            }
        elif reference_df is None:
            prediction_drift = {
                "skipped": True, "reason": "training reference unavailable", "rows": [], "overall_drifted": False,
            }
        else:
            try:
                from churnops.streaming.model_loader import load_scoring_model

                loaded = load_scoring_model(model_source)
                model = loaded.model
                reference_probs = get_reference_prediction_distribution(reference_df, model)
                prediction_drift = prediction_drift_report(
                    reference_probs,
                    current_df["churn_probability"].to_numpy(),
                    moderate_threshold=pred_cfg["moderate_threshold"],
                    significant_threshold=pred_cfg["significant_threshold"],
                    min_sample_size=pred_cfg["min_sample_size"],
                    histogram_buckets=pred_cfg["histogram_buckets"],
                    current_window=f"last {records_processed} predictions (source={source})",
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Prediction drift computation failed: %s", exc, exc_info=True)
                prediction_drift = {"skipped": True, "reason": str(exc), "rows": [], "overall_drifted": False}

        # ── 4. Data drift (needs a live raw-topic window) ───────────────────────
        data_cfg = cfg["drift"]["data"]
        if source != "kafka":
            data_drift = {
                "skipped": True,
                "reason": "data drift needs a raw-topic window; only available with --source kafka",
                "rows": [], "overall_drifted": False,
            }
        elif not data_cfg.get("enabled", True):
            data_drift = {"skipped": True, "reason": "disabled in configs/monitoring.yaml", "rows": [], "overall_drifted": False}
        elif reference_df is None:
            data_drift = {"skipped": True, "reason": "training reference unavailable", "rows": [], "overall_drifted": False}
        else:
            raw_df = fetch_raw_drift_window(
                cfg, bootstrap_servers=bootstrap,
                raw_topic=settings.kafka_topic_raw_customers,
                max_records=window,
            )
            if raw_df is None:
                data_drift = {
                    "skipped": True,
                    "reason": "raw window unavailable (Kafka unreachable or topic empty)",
                    "rows": [], "overall_drifted": False,
                }
            else:
                data_drift = data_drift_report(
                    reference_df, raw_df,
                    numeric_cols=data_cfg["numeric_cols"],
                    categorical_cols=data_cfg["categorical_cols"],
                    psi_threshold=data_cfg["psi_threshold"],
                    chi_square_alpha=data_cfg["chi_square_alpha"],
                    min_sample_size=data_cfg["min_sample_size"],
                    histogram_buckets=data_cfg["histogram_buckets"],
                    current_window=f"last {len(raw_df)} raw events",
                )
                if not dry_run and "Contract" in raw_df.columns:
                    from churnops.data.schema import ID_COL

                    contract_map = dict(zip(raw_df[ID_COL], raw_df["Contract"], strict=False))
                    n_backfilled = store.backfill_contract(session, contract_map)
                    result["contract_backfilled"] = n_backfilled

        # ── 5. Performance (only if ground truth present) ───────────────────────
        perf_cfg = cfg["performance"]
        if records_processed == 0:
            performance = {
                "has_ground_truth": False, "sample_size": 0, "metrics": {}, "degraded": False,
                "reason": "no records in window", "rows": [],
            }
        else:
            baseline = perf_cfg["baseline_roc_auc"]
            if perf_cfg.get("baseline_source") == "mlflow":
                baseline = _resolve_mlflow_baseline(model_version) or baseline
            performance = performance_report(
                current_df,
                baseline_roc_auc=baseline,
                degradation_tolerance=perf_cfg["degradation_tolerance"],
                min_labeled_sample=perf_cfg["min_labeled_sample"],
                model_version=model_version,
            )

        # ── 6. Persist metrics ────────────────────────────────────────────────────
        if not dry_run:
            store.write_drift_metrics(session, prediction_drift.get("rows", []) + data_drift.get("rows", []))
            store.write_performance_metrics(session, performance.get("rows", []))

        # ── 7. Alerts ──────────────────────────────────────────────────────────
        alert_candidates = evaluate_alert_rules(
            prediction_drift=prediction_drift,
            data_drift=data_drift,
            performance=performance,
            records_processed=records_processed,
            config=cfg["alerts"],
        )
        alerts_fired = fire_alerts(session, alert_candidates, cfg["alerts"], dry_run=dry_run)

        # ── 8. monitoring_runs summary row ────────────────────────────────────────
        drift_detected = bool(prediction_drift.get("overall_drifted") or data_drift.get("overall_drifted"))
        performance_degraded = bool(performance.get("degraded"))
        notes = (
            f"prediction_drift={'skipped' if prediction_drift.get('skipped') else prediction_drift.get('psi_band')}; "
            f"data_drift={'skipped: ' + str(data_drift.get('reason')) if data_drift.get('skipped') else 'computed'}; "
            f"performance={'no ground truth' if not performance.get('has_ground_truth') else performance.get('reason')}"
        )
        if not dry_run:
            store.write_monitoring_run(session, {
                "records_processed": records_processed,
                "drift_detected": drift_detected,
                "performance_degraded": performance_degraded,
                "alerts_fired": alerts_fired,
                "status": "ok",
                "notes": notes,
            })
            session.commit()
        else:
            session.rollback()

        result.update({
            "model_version": model_version,
            "prediction_drift": prediction_drift,
            "data_drift": data_drift,
            "performance": performance,
            "alerts_fired": alerts_fired,
            "alerts": alert_candidates,
            "drift_detected": drift_detected,
            "performance_degraded": performance_degraded,
            "notes": notes,
            "status": "ok",
        })
        return result

    except Exception:
        session.rollback()
        if not dry_run:
            try:
                store.write_monitoring_run(session, {
                    "records_processed": result.get("records_processed", 0),
                    "drift_detected": False,
                    "performance_degraded": False,
                    "alerts_fired": 0,
                    "status": "error",
                    "notes": "monitoring cycle raised — see logs",
                })
                session.commit()
            except Exception:  # noqa: BLE001
                session.rollback()
        raise
    finally:
        session.close()


# ── Console report ────────────────────────────────────────────────────────────

def render_report(result: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("=" * 68)
    lines.append("  MONITORING CYCLE REPORT")
    lines.append("=" * 68)
    lines.append(f"  run_ts            : {result['run_ts']}")
    lines.append(f"  source            : {result['source']}  (window={result['window']})")
    lines.append(f"  dry_run           : {result['dry_run']}")
    lines.append(f"  database          : {result.get('database_url_host')}")
    lines.append(f"  records_processed : {result.get('records_processed')}")
    lines.append(f"  records_inserted  : {result.get('records_inserted')}")
    lines.append(f"  model_version     : {result.get('model_version')}")
    lines.append("-" * 68)

    pd_ = result.get("prediction_drift", {})
    lines.append("  Prediction drift:")
    if pd_.get("skipped"):
        lines.append(f"    skipped — {pd_.get('reason')}")
    else:
        lines.append(f"    psi={pd_.get('psi'):.4f} ({pd_.get('psi_band')})  ks={pd_.get('ks_statistic'):.4f}  "
                      f"p={pd_.get('ks_pvalue'):.4f}  drifted={pd_.get('overall_drifted')}")

    dd = result.get("data_drift", {})
    lines.append("  Data drift:")
    if dd.get("skipped"):
        lines.append(f"    skipped — {dd.get('reason')}")
    else:
        for row in dd.get("rows", []):
            flag = "DRIFTED" if row["is_drifted"] else "ok"
            lines.append(
                f"    {row['feature_name']:<20} {row['statistic_name']:<10} "
                f"{row['statistic_value']:.4f}  ({flag})"
            )

    perf = result.get("performance", {})
    lines.append("  Performance:")
    if not perf.get("has_ground_truth"):
        lines.append(f"    skipped — {perf.get('reason')}")
    else:
        metrics_str = "  ".join(
            f"{k}={v:.4f}" for k, v in perf.get("metrics", {}).items() if v is not None
        )
        lines.append(f"    n={perf.get('sample_size')}  {metrics_str}  degraded={perf.get('degraded')}")

    lines.append("-" * 68)
    lines.append(f"  drift_detected       : {result.get('drift_detected')}")
    lines.append(f"  performance_degraded : {result.get('performance_degraded')}")
    lines.append(f"  alerts_fired         : {result.get('alerts_fired')}")
    for alert in result.get("alerts", []):
        lines.append(f"    [{alert['severity'].upper()}][{alert['category']}] {alert['message']}")
    lines.append("=" * 68)
    return "\n".join(lines)
