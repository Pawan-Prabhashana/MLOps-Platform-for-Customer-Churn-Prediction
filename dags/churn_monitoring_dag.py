# ruff: noqa: E402
"""churn_monitoring — every-15-min drift/performance monitoring DAG.

Pipeline:
    run_monitoring_cycle -> check_should_retrain --(short-circuit)--> trigger_retrain

Design philosophy
-----------------
* A thin wrapper around src/churnops/monitoring/collector.run_cycle — no
  monitoring logic lives here, only orchestration.
* run_monitoring_cycle ingests a bounded window of telco.churn.predictions,
  computes drift + performance, persists to the monitoring store, and
  evaluates alert rules (see configs/monitoring.yaml).
* check_should_retrain is a ShortCircuitOperator: retraining is only
  triggered when BOTH configs/monitoring.yaml retrain_trigger.enabled is
  true AND this cycle reported drift_detected=True. The flag defaults to
  OFF (opt-in) so drift alone never silently kicks off an unattended
  retrain — flip it on once you're happy with the retrain DAG's behavior.
* Connection strings: KAFKA_BOOTSTRAP_SERVERS / DATABASE_URL env vars (set in
  docker-compose) or churnops.config defaults, same as the other DAGs.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import timedelta
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "dags") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "dags"))

import yaml
from airflow import DAG
from airflow.decorators import task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.dates import days_ago
from airflow.utils.trigger_rule import TriggerRule
from utils.alerts import on_failure_callback

logger = logging.getLogger(__name__)


def _load_airflow_cfg() -> dict:
    with (_REPO_ROOT / "configs" / "airflow.yaml").open() as f:
        return yaml.safe_load(f)


def _load_monitoring_cfg() -> dict:
    with (_REPO_ROOT / "configs" / "monitoring.yaml").open() as f:
        return yaml.safe_load(f)


_CFG = _load_airflow_cfg()
_MON_CFG = _load_monitoring_cfg()

# Airflow Variables / env overrides (set in UI: Admin -> Variables):
#   CHURN_MONITORING_SCHEDULE   default from configs/airflow.yaml
#   CHURN_MONITORING_WINDOW     default from configs/airflow.yaml
_SCHEDULE = os.environ.get("CHURN_MONITORING_SCHEDULE", _CFG["monitoring_schedule"])
_WINDOW = int(os.environ.get("CHURN_MONITORING_WINDOW", _CFG["monitoring_window"]))
_TASK_TIMEOUT = timedelta(minutes=_CFG["task_timeout_minutes"])
_RETRIES = _CFG["task_retries"]
_RETRY_DELAY = timedelta(minutes=_CFG["task_retry_delay_minutes"])
_RETRAIN_DAG_ID = _MON_CFG.get("retrain_trigger", {}).get("dag_id", "churn_retrain")

_DEFAULT_ARGS = {
    "owner": "churnops",
    "retries": _RETRIES,
    "retry_delay": _RETRY_DELAY,
    "email_on_failure": _CFG["email_on_failure"],
    "email_on_retry": _CFG["email_on_retry"],
    "email": _CFG["email"],
}

# ── DAG ───────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="churn_monitoring",
    description="Every-15-min drift/performance monitoring: ingest -> compute -> persist -> alert",
    default_args=_DEFAULT_ARGS,
    schedule_interval=_SCHEDULE,
    start_date=days_ago(1),
    catchup=False,
    tags=["churnops", "monitoring", "mlops"],
    on_failure_callback=on_failure_callback,
    doc_md=__doc__,
) as dag:

    # ── Task 1: run_monitoring_cycle ──────────────────────────────────────────

    @task(task_id="run_monitoring_cycle", execution_timeout=_TASK_TIMEOUT)
    def run_monitoring_cycle() -> dict:
        """Run one full monitoring cycle via the collector (source=kafka)."""
        from churnops.monitoring.collector import run_cycle

        result = run_cycle(source="kafka", window=_WINDOW, dry_run=False)

        logger.info(
            "churn_monitoring cycle complete: records=%d drift_detected=%s "
            "performance_degraded=%s alerts_fired=%d",
            result.get("records_processed", 0),
            result.get("drift_detected"),
            result.get("performance_degraded"),
            result.get("alerts_fired", 0),
        )
        # Keep the XCom small — drop the heavy per-feature row lists.
        return {
            "records_processed": result.get("records_processed", 0),
            "drift_detected": result.get("drift_detected", False),
            "performance_degraded": result.get("performance_degraded", False),
            "alerts_fired": result.get("alerts_fired", 0),
            "notes": result.get("notes"),
        }

    # ── Task 2: check_should_retrain (short-circuit gate) ─────────────────────

    @task.short_circuit(task_id="check_should_retrain", execution_timeout=_TASK_TIMEOUT)
    def check_should_retrain(monitoring_result: dict) -> bool:
        """Gate the retrain trigger behind configs/monitoring.yaml retrain_trigger.enabled.

        Returns True (proceed to trigger_retrain) only when the flag is on
        AND this cycle detected drift. Defaults to False — opt-in only.
        """
        enabled = bool(_MON_CFG.get("retrain_trigger", {}).get("enabled", False))
        drift_detected = bool(monitoring_result.get("drift_detected", False))

        if not enabled:
            logger.info("Retrain trigger disabled (retrain_trigger.enabled=false) — skipping.")
            return False
        if not drift_detected:
            logger.info("No drift detected this cycle — skipping retrain trigger.")
            return False

        logger.info("Drift detected AND retrain_trigger enabled — triggering %s.", _RETRAIN_DAG_ID)
        return True

    # ── Task 3: trigger_retrain (only runs if the short-circuit passes) ───────

    trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id=_RETRAIN_DAG_ID,
        trigger_rule=TriggerRule.ALL_SUCCESS,
        execution_timeout=_TASK_TIMEOUT,
    )

    # ── Dependency wiring ─────────────────────────────────────────────────────

    t_monitor = run_monitoring_cycle()
    t_gate = check_should_retrain(t_monitor)
    t_monitor >> t_gate >> trigger_retrain
