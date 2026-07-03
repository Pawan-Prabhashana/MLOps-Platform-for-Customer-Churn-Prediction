"""Model Registry helpers: registration, alias management, promotion, and loading.

Uses the modern MLflow alias API (set_registered_model_alias /
get_model_version_by_alias) — NOT the deprecated stage transitions.

Alias conventions (from configs/model.yaml):
  staging    → most recently registered version
  production → best version by the configured promotion metric
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from churnops.tracking.mlflow_utils import (
    get_promotion_metric,
    get_registered_model_name,
    get_run_metric,
    load_mlflow_config,
    setup_tracking,
)

logger = logging.getLogger(__name__)


# ── Data class for version comparison ─────────────────────────────────────────

@dataclass
class VersionInfo:
    version: str
    run_id: str
    metric_value: float | None
    aliases: list[str]
    model_key: str | None

    def __str__(self) -> str:
        metric = f"{self.metric_value:.4f}" if self.metric_value is not None else "N/A"
        aliases = ", ".join(self.aliases) if self.aliases else "-"
        return (
            f"v{self.version:<4}  {metric:<8}  aliases=[{aliases}]  "
            f"model={self.model_key or '?'}  run={self.run_id[:8]}"
        )


# ── Ensure registered model exists ───────────────────────────────────────────

def ensure_registered_model(name: str | None = None, client: MlflowClient | None = None) -> None:
    """Create the registered model entry if it doesn't exist yet."""
    c = client or MlflowClient()
    model_name = name or get_registered_model_name()
    try:
        c.get_registered_model(model_name)
    except MlflowException:
        c.create_registered_model(
            model_name,
            description="Churn prediction sklearn Pipeline (churnops project)",
        )
        logger.info("Created registered model: %s", model_name)


# ── Register a model from a run ───────────────────────────────────────────────

def register_model(
    run_id: str,
    model_artifact_path: str = "model",
    model_name: str | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    """Register the logged model from `run_id` and return the version number.

    Args:
        run_id:               MLflow run ID that contains the logged model.
        model_artifact_path:  Path within the run's artifacts (default "model").
        model_name:           Registry name (default from config).
        tags:                 Optional tags for the model version.

    Returns:
        The registered version string (e.g. "3").
    """
    c = MlflowClient()
    model_name = model_name or get_registered_model_name()
    ensure_registered_model(model_name, client=c)

    model_uri = f"runs:/{run_id}/{model_artifact_path}"
    mv = mlflow.register_model(model_uri, model_name, tags=tags)
    logger.info("Registered %s version %s from run %s", model_name, mv.version, run_id[:8])
    return mv.version


# ── Alias management ──────────────────────────────────────────────────────────

def set_alias(version: str, alias: str, model_name: str | None = None) -> None:
    """Assign `alias` to model version `version`."""
    c = MlflowClient()
    name = model_name or get_registered_model_name()
    c.set_registered_model_alias(name, alias, version)
    logger.info("Alias '%s' → %s v%s", alias, name, version)


def get_alias(alias: str, model_name: str | None = None) -> str | None:
    """Return the version number currently assigned to `alias`, or None."""
    c = MlflowClient()
    name = model_name or get_registered_model_name()
    try:
        mv = c.get_model_version_by_alias(name, alias)
        return mv.version
    except MlflowException:
        return None


# ── List all versions with metrics ───────────────────────────────────────────

def list_versions(
    model_name: str | None = None,
    metric_name: str | None = None,
    client: MlflowClient | None = None,
) -> list[VersionInfo]:
    """Return VersionInfo for every registered version, sorted by version number."""
    c = client or MlflowClient()
    name = model_name or get_registered_model_name()
    metric = metric_name or get_promotion_metric()

    try:
        rm = c.get_registered_model(name)
    except MlflowException:
        return []

    versions: list[VersionInfo] = []
    for mv in rm.latest_versions or []:
        pass  # latest_versions only gives one per stage (deprecated); use search instead

    # Build alias→version reverse map from the registered model object
    alias_to_version: dict[str, str] = getattr(rm, "aliases", {}) or {}
    version_to_aliases: dict[str, list[str]] = {}
    for alias, ver in alias_to_version.items():
        version_to_aliases.setdefault(ver, []).append(alias)

    # search_model_versions gives all
    for mv in c.search_model_versions(f"name='{name}'"):
        metric_val = get_run_metric(mv.run_id, metric, client=c)
        versions.append(
            VersionInfo(
                version=mv.version,
                run_id=mv.run_id,
                metric_value=metric_val,
                aliases=version_to_aliases.get(mv.version, []),
                model_key=mv.tags.get("model_key") if mv.tags else None,
            )
        )

    versions.sort(key=lambda v: int(v.version))
    return versions


# ── Promotion logic ───────────────────────────────────────────────────────────

def promote_best(
    model_name: str | None = None,
    metric_name: str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> tuple[str | None, str | None]:
    """Promote the best-scoring version to production alias.

    Args:
        model_name:   Registry model name (default from config).
        metric_name:  Metric to compare (default from config).
        force:        Promote even if it's not better than current production.
        dry_run:      Print what would happen but don't actually change aliases.

    Returns:
        (new_production_version, reason_string)
    """
    cfg = load_mlflow_config()
    name = model_name or get_registered_model_name()
    metric = metric_name or get_promotion_metric()
    prod_alias = cfg["aliases"]["production"]
    staging_alias = cfg["aliases"]["staging"]

    versions = list_versions(name, metric)
    if not versions:
        return None, "No versions found."

    # Filter to versions that have a metric value
    scored = [v for v in versions if v.metric_value is not None]
    if not scored:
        return None, "No versions have the promotion metric logged."

    best = max(scored, key=lambda v: v.metric_value)  # type: ignore[arg-type]
    current_prod_ver = get_alias(prod_alias, name)
    latest_ver = versions[-1].version

    reason_parts = [f"Best version: v{best.version} ({metric}={best.metric_value:.4f})"]

    if current_prod_ver:
        current_prod_info = next((v for v in versions if v.version == current_prod_ver), None)
        current_metric = current_prod_info.metric_value if current_prod_info else None
        if current_metric is not None:
            reason_parts.append(
                f"Current production v{current_prod_ver} has {metric}={current_metric:.4f}"
            )
            if best.metric_value <= current_metric and not force:  # type: ignore[operator]
                msg = (
                    f"Best v{best.version} ({best.metric_value:.4f}) does not beat "  # type: ignore[str-bytes-safe]
                    f"current production v{current_prod_ver} ({current_metric:.4f}). "
                    "Use --force to promote anyway."
                )
                return None, msg

    reason = " | ".join(reason_parts)
    if dry_run:
        logger.info("[DRY RUN] Would set production → v%s (%s)", best.version, reason)
        return best.version, reason

    set_alias(best.version, prod_alias, name)
    set_alias(latest_ver, staging_alias, name)
    return best.version, reason


# ── Load by alias ─────────────────────────────────────────────────────────────

def load_production_model(
    model_name: str | None = None,
    alias: str | None = None,
    tracking_uri: str | None = None,
) -> Any:
    """Load the sklearn Pipeline from the registry by alias.

    Falls back to loading the local joblib artifact if the registry is
    unreachable or the alias is not set.

    Args:
        model_name: Registry name (default from config).
        alias:      Alias to load (default: "production").
        tracking_uri: Override tracking URI (default from config).

    Returns:
        Fitted sklearn Pipeline.
    """
    cfg = load_mlflow_config()
    name = model_name or get_registered_model_name()
    resolved_alias = alias or cfg["aliases"]["production"]

    try:
        setup_tracking(tracking_uri)
        model_uri = f"models:/{name}@{resolved_alias}"
        logger.info("Loading model from registry: %s", model_uri)
        return mlflow.sklearn.load_model(model_uri)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Registry load failed (%s). Falling back to local joblib artifact.", exc
        )
        from churnops.models.persistence import load_pipeline
        return load_pipeline()
