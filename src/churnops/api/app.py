"""FastAPI application factory with lifespan model loading.

The model is loaded ONCE during the lifespan startup and stored on
``app.state.model_state``. If loading fails, the app still starts and
``/health`` reports the degraded state (HTTP 503) so ECS health checks can
surface the problem without crash-looping the container.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from churnops.api.routes import router
from churnops.logging import setup_logging

setup_logging()
logger = logging.getLogger("churnops.api")

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_API_YAML = _REPO_ROOT / "configs" / "api.yaml"


def _load_api_config() -> dict:
    with _API_YAML.open() as f:
        return yaml.safe_load(f)


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the scoring model at startup; release resources on shutdown."""
    cfg = _load_api_config()
    model_source = os.environ.get("MODEL_SOURCE") or cfg.get("model", {}).get("source", "registry")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

    logger.info(
        "API starting up — model_source=%s tracking_uri=%s",
        model_source,
        tracking_uri or "(from config)",
    )

    start = time.perf_counter()
    try:
        from churnops.streaming.model_loader import load_scoring_model

        loaded = load_scoring_model(
            model_source,
            tracking_uri=tracking_uri,
            use_cache=False,  # fresh load; we manage state ourselves
        )
        elapsed = time.perf_counter() - start

        # Extract version from the source string when it's a registry load.
        version: str | None = None
        if "registry:" in loaded.source:
            # source looks like "registry:models:/churn-classifier@production"
            try:
                from mlflow import MlflowClient

                from churnops.tracking.mlflow_utils import (
                    get_registered_model_name,
                    load_mlflow_config,
                )
                mlflow_cfg = load_mlflow_config()
                alias = mlflow_cfg["aliases"]["production"]
                name = get_registered_model_name()
                client = MlflowClient()
                mv = client.get_model_version_by_alias(name, alias)
                version = mv.version
            except Exception:  # noqa: BLE001
                pass

        app.state.model_state = {
            "model_loaded": True,
            "model": loaded.model,
            "source": loaded.source,
            "version": version,
            "load_time": datetime.now(UTC).isoformat(),
            "load_elapsed_s": round(elapsed, 3),
            "startup_ts": time.perf_counter(),
        }
        logger.info(
            "Model loaded in %.2fs — source=%s version=%s",
            elapsed,
            loaded.source,
            version or "n/a",
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Model load FAILED at startup: %s", exc)
        app.state.model_state = {
            "model_loaded": False,
            "model": None,
            "source": None,
            "version": None,
            "load_time": None,
            "startup_ts": time.perf_counter(),
            "load_error": str(exc),
        }

    yield  # app is running

    logger.info("API shutting down.")


# ── Application factory ───────────────────────────────────────────────────────


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    cfg = _load_api_config()

    app = FastAPI(
        title="ChurnOps Prediction API",
        description=(
            "Real-time Telco customer churn scoring API. "
            "Loads the production sklearn model from the MLflow registry "
            "(joblib fallback). "
            "POST to /predict for single-record scoring or /predict/batch "
            "for vectorized batch scoring."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    cors_cfg = cfg.get("cors", {})
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_cfg.get("allow_origins", ["*"]),
        allow_credentials=cors_cfg.get("allow_credentials", False),
        allow_methods=cors_cfg.get("allow_methods", ["*"]),
        allow_headers=cors_cfg.get("allow_headers", ["*"]),
    )

    app.include_router(router)
    return app


# Module-level app instance (used by uvicorn as churnops.api.app:app)
app = create_app()


def _uptime(state: dict) -> float | None:
    ts = state.get("startup_ts")
    return round(time.perf_counter() - ts, 1) if ts else None


# Patch uptime into health responses dynamically
@app.middleware("http")
async def _inject_uptime(request, call_next):
    state = getattr(request.app.state, "model_state", None)
    if state is not None:
        state["uptime_seconds"] = _uptime(state)
    return await call_next(request)
