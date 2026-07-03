"""Load the churn scoring model ONCE at consumer startup.

Default source is the MLflow registry production alias
(``models:/churn-classifier@production``). If the registry / tracking server is
unreachable, we transparently fall back to the local joblib pipeline artifact.

IMPORTANT: this loads the **sklearn** production model. The Spark MLlib model
cannot be reloaded in-process on this Python 3.14 interpreter (bundled
cloudpickle recursion bug), so it is deliberately *not* wired as a source here.

The loaded object is a fitted sklearn Pipeline whose first stage is the
ColumnTransformer — so it accepts raw-cleaned feature rows and returns
predictions directly (no separate preprocessing needed by the consumer).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Process-wide cache so the model is only materialised once even if the loader
# is called from multiple places.
_CACHE: dict[str, LoadedModel] = {}


@dataclass
class LoadedModel:
    """A loaded scoring model plus a human-readable description of its source."""

    model: Any
    source: str  # e.g. "registry:models:/churn-classifier@production" or "joblib"

    def predict_proba_churn(self, features: Any) -> Any:
        """Return P(churn=1) for each row of a feature DataFrame."""
        return self.model.predict_proba(features)[:, 1]


def _load_from_registry(
    model_name: str | None,
    alias: str | None,
    tracking_uri: str | None,
) -> LoadedModel:
    """Load the sklearn model from the MLflow registry by alias.

    Raises on any failure so the caller can decide whether to fall back.
    """
    import mlflow.sklearn

    from churnops.tracking.mlflow_utils import (
        get_registered_model_name,
        load_mlflow_config,
        setup_tracking,
    )

    cfg = load_mlflow_config()
    name = model_name or get_registered_model_name()
    resolved_alias = alias or cfg["aliases"]["production"]

    setup_tracking(tracking_uri)
    model_uri = f"models:/{name}@{resolved_alias}"
    logger.info("Loading scoring model from registry: %s", model_uri)
    model = mlflow.sklearn.load_model(model_uri)
    return LoadedModel(model=model, source=f"registry:{model_uri}")


def _load_from_joblib() -> LoadedModel:
    """Load the local joblib pipeline artifact."""
    from churnops.models.persistence import load_pipeline

    model = load_pipeline()
    logger.info("Loaded scoring model from local joblib artifact.")
    return LoadedModel(model=model, source="joblib")


def load_scoring_model(
    source: str = "registry",
    *,
    model_name: str | None = None,
    alias: str | None = None,
    tracking_uri: str | None = None,
    use_cache: bool = True,
) -> LoadedModel:
    """Load the churn scoring model once and cache it.

    Args:
        source:       "registry" (default) or "joblib".
        model_name:   Override the registered model name.
        alias:        Override the registry alias (default "production").
        tracking_uri: Override the MLflow tracking URI.
        use_cache:    Reuse a previously loaded model for the same source.

    Returns:
        A :class:`LoadedModel` wrapping the fitted pipeline + its source string.
    """
    if use_cache and source in _CACHE:
        return _CACHE[source]

    if source == "joblib":
        loaded = _load_from_joblib()
    elif source == "registry":
        try:
            loaded = _load_from_registry(model_name, alias, tracking_uri)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Registry load failed (%s). Falling back to local joblib artifact.",
                exc,
            )
            loaded = _load_from_joblib()
            loaded.source = f"joblib (registry fallback: {type(exc).__name__})"
    else:
        raise ValueError(f"Unknown model source '{source}'. Use 'registry' or 'joblib'.")

    if use_cache:
        _CACHE[source] = loaded
    logger.info("Scoring model ready — source=%s", loaded.source)
    return loaded


def clear_cache() -> None:
    """Clear the model cache (useful in tests)."""
    _CACHE.clear()
