"""Route handlers for the churn prediction API."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from churnops.api.deps import get_model_state
from churnops.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    BatchSummary,
    CustomerRecord,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from churnops.data.schema import ALL_FEATURE_COLS, ID_COL

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_API_YAML = _REPO_ROOT / "configs" / "api.yaml"


def _load_api_config() -> dict:
    with _API_YAML.open() as f:
        return yaml.safe_load(f)


router = APIRouter()


# ── /health ───────────────────────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health(request: Request):
    """Liveness + readiness probe.

    Always returns 200 when the process is alive. Returns 503 when the model
    failed to load at startup so ECS/ALB health checks can distinguish a
    healthy pod from a degraded one.
    """
    state = getattr(request.app.state, "model_state", None)
    loaded = bool(state and state.get("model_loaded"))

    response = HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_source=state.get("source") if state else None,
        model_version=state.get("version") if state else None,
        uptime_seconds=state.get("uptime_seconds") if state else None,
    )

    if not loaded:
        # Return a 503 so load-balancer health checks can detect the bad state.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=response.model_dump(),
        )
    return response


# ── /model-info ───────────────────────────────────────────────────────────────


@router.get("/model-info", response_model=ModelInfoResponse, tags=["ops"])
def model_info(state: dict = Depends(get_model_state)):
    """Return metadata about the loaded model: source, version, threshold, features."""
    cfg = _load_api_config()
    threshold = float(cfg.get("model", {}).get("threshold", 0.5))

    source: str = state.get("source", "unknown")
    version = state.get("version")
    registered_name = None
    alias = None

    if source.startswith("registry:"):
        # "registry:models:/churn-classifier@production"
        uri_part = source[len("registry:"):]
        parts = uri_part.split("/")
        if len(parts) >= 3:
            registered_name = parts[2].split("@")[0]
            alias_parts = parts[2].split("@")
            alias = alias_parts[1] if len(alias_parts) > 1 else None

    return ModelInfoResponse(
        model_source=source,
        model_version=version,
        registered_model_name=registered_name,
        alias=alias,
        threshold=threshold,
        feature_schema_version="1.0.0",
        feature_columns=ALL_FEATURE_COLS,
        load_time=state.get("load_time"),
    )


# ── Shared scoring helper ─────────────────────────────────────────────────────


def _build_feature_df(records: list[CustomerRecord]) -> pd.DataFrame:
    """Convert a list of CustomerRecords into a model-ready feature DataFrame.

    Applies the same light cleaning the training pipeline expects:
    - Yes/No binary strings → 1/0 integers (Partner, Dependents, etc.)
    - Numeric dtype coercion (TotalCharges blank → 0.0, etc.)

    We reuse ``clean_telco`` by injecting a synthetic "Churn" target column
    (value "No") before cleaning, then dropping it — exactly what the Kafka
    consumer does for records that don't carry ground truth.
    """
    from churnops.data.clean import clean_telco
    from churnops.data.schema import TARGET_RAW

    rows = []
    for rec in records:
        row: dict[str, Any] = rec.model_dump(exclude={"customerID"})
        cid = rec.customerID
        if cid is not None:
            row[ID_COL] = cid
        # Inject synthetic target so clean_telco can run unchanged.
        row[TARGET_RAW] = "No"
        rows.append(row)

    df = pd.DataFrame(rows)
    cleaned = clean_telco(df)
    # Drop the encoded target column that clean_telco added; keep features only.
    from churnops.data.schema import TARGET_COL
    cleaned = cleaned.drop(columns=[TARGET_COL], errors="ignore")
    return cleaned


def _score_records(
    records: list[CustomerRecord],
    model: Any,
    threshold: float,
    model_source: str,
    model_version: str | None,
) -> list[PredictionResponse]:
    """Run model inference and return PredictionResponse objects."""
    df = _build_feature_df(records)

    # Use the existing inference helper (handles missing cols, dtype coercion etc.)
    from churnops.models.inference import predict

    results = predict(df, pipeline=model)

    responses = []
    for rec, result in zip(records, results):
        proba = float(result.churn_probability)
        pred = "Yes" if proba >= threshold else "No"
        responses.append(
            PredictionResponse(
                customerID=rec.customerID,
                churn_probability=round(proba, 4),
                prediction=pred,
                threshold=threshold,
                model_source=model_source,
                model_version=model_version,
            )
        )
    return responses


# ── POST /predict ─────────────────────────────────────────────────────────────


@router.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict_single(
    record: CustomerRecord,
    threshold: float | None = Query(None, ge=0.0, le=1.0, description="Override the default threshold"),
    state: dict = Depends(get_model_state),
):
    """Score a single customer record and return a churn prediction.

    The pipeline contains all preprocessing so raw field values are accepted
    directly. Unseen categorical values are handled by the pipeline's
    ``handle_unknown='ignore'`` setting — they do not cause a 500.
    """
    cfg = _load_api_config()
    effective_threshold = threshold if threshold is not None else float(cfg.get("model", {}).get("threshold", 0.5))

    try:
        results = _score_records(
            [record],
            state["model"],
            effective_threshold,
            state.get("source", "unknown"),
            state.get("version"),
        )
        return results[0]
    except Exception as exc:
        logger.exception("Scoring error for customerID=%s", record.customerID)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scoring failed: {exc}",
        ) from exc


# ── POST /predict/batch ───────────────────────────────────────────────────────


@router.post("/predict/batch", response_model=BatchPredictResponse, tags=["inference"])
def predict_batch(
    body: BatchPredictRequest,
    state: dict = Depends(get_model_state),
):
    """Score a batch of customer records in a single vectorized call.

    Rejects batches larger than the configured ``batch.max_records`` with HTTP 413.
    """
    cfg = _load_api_config()
    max_records = int(cfg.get("batch", {}).get("max_records", 500))

    if len(body.records) > max_records:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Batch has {len(body.records)} records; maximum is {max_records}.",
        )

    effective_threshold = body.threshold if body.threshold is not None else float(cfg.get("model", {}).get("threshold", 0.5))

    try:
        predictions = _score_records(
            body.records,
            state["model"],
            effective_threshold,
            state.get("source", "unknown"),
            state.get("version"),
        )
    except Exception as exc:
        logger.exception("Batch scoring error (%d records)", len(body.records))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch scoring failed: {exc}",
        ) from exc

    n = len(predictions)
    n_yes = sum(1 for p in predictions if p.prediction == "Yes")
    mean_prob = sum(p.churn_probability for p in predictions) / n

    return BatchPredictResponse(
        predictions=predictions,
        count=n,
        summary=BatchSummary(
            churn_rate=round(n_yes / n, 4),
            mean_probability=round(mean_prob, 4),
        ),
    )
