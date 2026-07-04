"""Threshold checks → alerting hooks.

Each fired alert is (a) written to the `alerts` table and (b) passed to every
configured hook. The plain logger hook always runs; Slack/SMTP are stubs —
each one line to wire up once credentials exist. A failing hook is caught
and logged; it never crashes the monitoring run.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from sqlalchemy.orm import Session

from churnops.monitoring import store

logger = logging.getLogger(__name__)


# ── Severity ──────────────────────────────────────────────────────────────────

def severity_for_breach(
    value: float,
    threshold: float,
    critical_multiplier: float = 2.0,
    info_multiplier: float = 1.2,
) -> str:
    """Scale severity by how far past `threshold` `value` is.

    "info" just past the line (< threshold * info_multiplier), "warning" past
    that, "critical" once it crosses threshold * critical_multiplier.
    Callers only invoke this after confirming the rule was breached at all
    (value >= threshold), so every call returns at least "info".
    """
    if threshold <= 0:
        return "warning"
    ratio = value / threshold
    if ratio >= critical_multiplier:
        return "critical"
    if ratio >= info_multiplier:
        return "warning"
    return "info"


# ── Rule evaluation ────────────────────────────────────────────────────────────

def evaluate_alert_rules(
    *,
    prediction_drift: dict[str, Any] | None,
    data_drift: dict[str, Any] | None,
    performance: dict[str, Any] | None,
    records_processed: int,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return a list of alert dicts (not yet persisted) for every rule breach."""
    alerts: list[dict[str, Any]] = []
    severity_cfg = config.get("severity", {})
    critical_mult = severity_cfg.get("critical_multiplier", 2.0)
    info_mult = severity_cfg.get("info_multiplier", 1.2)

    def _severity(value: float, threshold: float) -> str:
        return severity_for_breach(value, threshold, critical_mult, info_mult)

    # ── Prediction drift: overall PSI ────────────────────────────────────────
    if prediction_drift and not prediction_drift.get("skipped"):
        for row in prediction_drift.get("rows", []):
            if row["statistic_name"] == "psi" and row["is_drifted"]:
                alerts.append({
                    "severity": _severity(row["statistic_value"], row["threshold"]),
                    "category": "drift",
                    "message": (
                        f"Prediction drift: overall PSI={row['statistic_value']:.4f} "
                        f">= threshold={row['threshold']:.4f} "
                        f"({prediction_drift.get('psi_band', 'significant')} drift)"
                    ),
                    "metric_value": row["statistic_value"],
                    "threshold": row["threshold"],
                })

    # ── Data drift: any monitored feature ────────────────────────────────────
    if data_drift and not data_drift.get("skipped"):
        for row in data_drift.get("rows", []):
            if row["is_drifted"]:
                alerts.append({
                    "severity": _severity(row["statistic_value"], row["threshold"]),
                    "category": "drift",
                    "message": (
                        f"Data drift on feature '{row['feature_name']}': "
                        f"{row['statistic_name']}={row['statistic_value']:.4f} "
                        f">= threshold={row['threshold']:.4f}"
                    ),
                    "metric_value": row["statistic_value"],
                    "threshold": row["threshold"],
                })

    # ── Performance degradation ───────────────────────────────────────────────
    if performance and performance.get("has_ground_truth") and performance.get("degraded"):
        roc_auc = performance["metrics"].get("roc_auc")
        baseline = performance.get("baseline_roc_auc")
        tolerance = performance.get("degradation_tolerance") or 0.05
        # Scale severity by how many "tolerance units" past the acceptable
        # floor (baseline - tolerance) roc_auc has fallen — 1.0 right at the
        # floor itself (barely degraded), rising from there. A flat
        # baseline-relative ratio would under-rate a severe collapse (e.g.
        # 0.80 -> 0.40 is catastrophic, not merely "past threshold").
        if baseline is not None and roc_auc is not None:
            floor = baseline - tolerance
            deficit_ratio = 1.0 + max(0.0, floor - roc_auc) / tolerance
        else:
            deficit_ratio = critical_mult
        alerts.append({
            "severity": severity_for_breach(deficit_ratio, 1.0, critical_mult, info_mult),
            "category": "performance",
            "message": f"Model performance degraded: {performance.get('reason')}",
            "metric_value": roc_auc,
            "threshold": baseline,
        })

    # ── Volume anomaly (pipeline stalled or spiking) ─────────────────────────
    vol_cfg = config.get("volume", {})
    lo = vol_cfg.get("min_records_per_window")
    hi = vol_cfg.get("max_records_per_window")
    if lo is not None and records_processed < lo:
        # Inverted breach: severity should rise the FURTHER BELOW lo we are
        # (0 records = worst case), not above it.
        deficit_ratio = 1.0 + (lo - records_processed) / lo if lo else critical_mult
        alerts.append({
            "severity": severity_for_breach(deficit_ratio, 1.0, critical_mult, info_mult),
            "category": "volume",
            "message": (
                f"Prediction volume anomalously LOW: {records_processed} records "
                f"this window (< min={lo}) — pipeline may have stalled."
            ),
            "metric_value": float(records_processed),
            "threshold": float(lo),
        })
    if hi is not None and records_processed > hi:
        alerts.append({
            "severity": _severity(float(records_processed), float(hi)),
            "category": "volume",
            "message": (
                f"Prediction volume anomalously HIGH: {records_processed} records "
                f"this window (> max={hi})."
            ),
            "metric_value": float(records_processed),
            "threshold": float(hi),
        })

    return alerts


# ── Hooks ─────────────────────────────────────────────────────────────────────

def _logger_hook(alert: dict[str, Any]) -> None:
    """Always-on sink: structured log line."""
    logger.warning(
        "[ALERT] severity=%s category=%s message=%s",
        alert["severity"], alert["category"], alert["message"],
    )


def _slack_hook(alert: dict[str, Any], webhook_env: str = "SLACK_WEBHOOK_URL") -> None:
    """Slack webhook stub. To enable: set SLACK_WEBHOOK_URL in .env."""
    webhook_url = os.environ.get(webhook_env)
    if not webhook_url:
        return
    import urllib.request

    payload = f'{{"text": "[{alert["severity"].upper()}][{alert["category"]}] {alert["message"]}"}}'
    req = urllib.request.Request(
        webhook_url, data=payload.encode("utf-8"), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10):
        pass


def _smtp_hook(alert: dict[str, Any], enabled: bool = False) -> None:
    """SMTP email stub. To enable: set smtp_enabled: true in configs/monitoring.yaml
    and configure smtplib/credentials here.
    """
    if not enabled:
        return
    # import smtplib
    # ... send alert["message"] via configured SMTP server ...


def run_hooks(alert: dict[str, Any], config: dict[str, Any]) -> None:
    """Call every configured hook for one alert. Never raises."""
    hooks_cfg = config.get("hooks", {})

    for hook, kwargs in (
        (_logger_hook, {}),
        (_slack_hook, {"webhook_env": hooks_cfg.get("slack_webhook_env", "SLACK_WEBHOOK_URL")}),
        (_smtp_hook, {"enabled": hooks_cfg.get("smtp_enabled", False)}),
    ):
        try:
            hook(alert, **kwargs)
        except Exception:  # noqa: BLE001
            logger.warning("Alert hook %s failed (non-fatal)", hook.__name__, exc_info=True)


# ── Fire (persist + hook) ─────────────────────────────────────────────────────

def fire_alerts(
    session: Session,
    alerts: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    dry_run: bool = False,
) -> int:
    """Persist each alert and run its hooks. Returns the count fired."""
    for alert in alerts:
        if not dry_run:
            store.write_alert(session, {**alert, "acknowledged": False})
        run_hooks(alert, config)
    return len(alerts)
