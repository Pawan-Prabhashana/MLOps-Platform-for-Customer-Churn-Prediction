"""Monitoring layer: drift detection, performance tracking, prediction store, alerting.

Writes to the same `predictions` / `drift_metrics` / `performance_metrics` /
`monitoring_runs` / `alerts` tables that the business dashboard (next part)
reads from. See db.py for the schema and store.py for read/write helpers.
"""
