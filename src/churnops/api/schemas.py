"""Pydantic v2 request/response schemas for the churn prediction API.

Field lists are derived from schema.py (the single source of truth for column
groupings) so the API can never drift from the model's expected feature set.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from churnops.data.schema import (
    BINARY_YN_COLS,
    CATEGORICAL_COLS,
    INTEGER_BINARY_COLS,
    NUMERIC_COLS,
)

# ── Request models ─────────────────────────────────────────────────────────────


class CustomerRecord(BaseModel):
    """Raw customer-record fields that the pipeline accepts directly.

    Types mirror the original dataset dtypes (pre-cleaning):
      - Numeric fields:  tenure (int), MonthlyCharges / TotalCharges (float).
      - Integer binary:  SeniorCitizen (int, 0 or 1).
      - Yes/No strings:  Partner, Dependents, PhoneService, PaperlessBilling.
      - Categoricals:    gender, MultipleLines, InternetService, …
      - customerID:      optional — echoed back in the response if provided.

    Heavy cleaning (Yes/No → 1/0, dtype coercion) is handled by the pipeline's
    ColumnTransformer, so raw strings are passed in as-is.
    """

    # ── Identifier (optional) ─────────────────────────────────────────────────
    customerID: str | None = Field(None, examples=["7590-VHVEG"])

    # ── Numeric features ──────────────────────────────────────────────────────
    tenure: int = Field(..., ge=0, examples=[1])
    MonthlyCharges: float = Field(..., ge=0.0, examples=[29.85])
    TotalCharges: float = Field(..., ge=0.0, examples=[29.85])

    # ── Integer binary ────────────────────────────────────────────────────────
    SeniorCitizen: int = Field(..., ge=0, le=1, examples=[0])

    # ── Yes/No binary strings ─────────────────────────────────────────────────
    Partner: str = Field(..., examples=["Yes"])
    Dependents: str = Field(..., examples=["No"])
    PhoneService: str = Field(..., examples=["No"])
    PaperlessBilling: str = Field(..., examples=["Yes"])

    # ── Categorical strings ───────────────────────────────────────────────────
    gender: str = Field(..., examples=["Female"])
    MultipleLines: str = Field(..., examples=["No phone service"])
    InternetService: str = Field(..., examples=["DSL"])
    OnlineSecurity: str = Field(..., examples=["No"])
    OnlineBackup: str = Field(..., examples=["Yes"])
    DeviceProtection: str = Field(..., examples=["No"])
    TechSupport: str = Field(..., examples=["No"])
    StreamingTV: str = Field(..., examples=["No"])
    StreamingMovies: str = Field(..., examples=["No"])
    Contract: str = Field(..., examples=["Month-to-month"])
    PaymentMethod: str = Field(..., examples=["Electronic check"])

    model_config = {"extra": "ignore"}  # silently drop unknown fields (incl. Churn)


# Sanity-check that CustomerRecord covers all expected feature columns.
_expected = set(NUMERIC_COLS + INTEGER_BINARY_COLS + BINARY_YN_COLS + CATEGORICAL_COLS)
_declared = set(CustomerRecord.model_fields) - {"customerID"}
assert _declared == _expected, (
    f"CustomerRecord fields out of sync with schema.py!\n"
    f"  missing from schema: {_declared - _expected}\n"
    f"  missing from model:  {_expected - _declared}"
)


class BatchPredictRequest(BaseModel):
    records: list[CustomerRecord] = Field(..., min_length=1)
    threshold: float | None = Field(None, ge=0.0, le=1.0, description="Per-request threshold override")

    @model_validator(mode="after")
    def _check_batch_size(self) -> BatchPredictRequest:
        # Actual max enforced in the route (reads from config); this is a
        # reasonable hard ceiling to prevent accidentally huge payloads.
        if len(self.records) > 5000:
            raise ValueError("Batch exceeds the hard ceiling of 5000 records.")
        return self


# ── Response models ────────────────────────────────────────────────────────────


class PredictionResponse(BaseModel):
    customerID: str | None = None
    churn_probability: Annotated[float, Field(ge=0.0, le=1.0)]
    prediction: str  # "Yes" or "No"
    threshold: float
    model_source: str
    model_version: str | None = None


class BatchSummary(BaseModel):
    churn_rate: float   # fraction (0–1)
    mean_probability: float


class BatchPredictResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int
    summary: BatchSummary


class HealthResponse(BaseModel):
    status: str           # "ok" | "degraded"
    model_loaded: bool
    model_source: str | None = None
    model_version: str | None = None
    uptime_seconds: float | None = None


class ModelInfoResponse(BaseModel):
    model_source: str
    model_version: str | None = None
    registered_model_name: str | None = None
    alias: str | None = None
    threshold: float
    feature_schema_version: str
    feature_columns: list[str]
    load_time: str | None = None
