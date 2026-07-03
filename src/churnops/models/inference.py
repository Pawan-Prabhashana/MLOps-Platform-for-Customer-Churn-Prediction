"""Inference helpers — run a loaded pipeline on raw cleaned-style records.

Accepts input as:
  - a single dict (one row)
  - a list of dicts (multiple rows)
  - a pandas DataFrame

The input must contain the same columns as the cleaned dataset (minus the
target column, which is obviously absent at inference time).  If extra columns
are present they are silently ignored.  If known columns are missing they are
filled with sensible defaults so the request doesn't crash (though callers
should ideally supply complete rows).

Returns a list of InferenceResult named-tuples with:
  - customer_id: str | None
  - prediction: int   (0 = No churn, 1 = Churn)
  - label: str        ("Yes" / "No")
  - churn_probability: float
"""

from __future__ import annotations

import logging
from typing import Any, NamedTuple

import pandas as pd
from sklearn.pipeline import Pipeline

from churnops.data.schema import (
    ALL_FEATURE_COLS,
    BINARY_YN_COLS,
    CATEGORICAL_COLS,
    ID_COL,
    INTEGER_BINARY_COLS,
    NUMERIC_COLS,
    TARGET_COL,
)
from churnops.models.persistence import load_pipeline

logger = logging.getLogger(__name__)

# Default fill values used when a field is absent at inference time
_NUMERIC_DEFAULT = 0.0
_BINARY_DEFAULT = 0
_CATEGORICAL_DEFAULT = "No"


class InferenceResult(NamedTuple):
    customer_id: str | None
    prediction: int
    label: str
    churn_probability: float


def _coerce_to_frame(records: Any) -> pd.DataFrame:
    """Convert dict / list-of-dicts / DataFrame to a consistent DataFrame."""
    if isinstance(records, pd.DataFrame):
        df = records.copy()
    elif isinstance(records, dict):
        df = pd.DataFrame([records])
    elif isinstance(records, list):
        df = pd.DataFrame(records)
    else:
        raise TypeError(
            f"records must be dict, list[dict], or DataFrame; got {type(records)}"
        )
    return df


def _fill_missing_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Add any missing feature columns with safe default values."""
    for col in NUMERIC_COLS:
        if col not in df.columns:
            logger.warning("Missing numeric column '%s'; filling with %s", col, _NUMERIC_DEFAULT)
            df[col] = _NUMERIC_DEFAULT
    for col in INTEGER_BINARY_COLS + BINARY_YN_COLS:
        if col not in df.columns:
            logger.warning("Missing binary column '%s'; filling with %s", col, _BINARY_DEFAULT)
            df[col] = _BINARY_DEFAULT
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            logger.warning(
                "Missing categorical column '%s'; filling with '%s'",
                col,
                _CATEGORICAL_DEFAULT,
            )
            df[col] = _CATEGORICAL_DEFAULT
    return df


def predict(
    records: Any,
    pipeline: Pipeline | None = None,
) -> list[InferenceResult]:
    """Run the pipeline on records and return InferenceResult list.

    Args:
        records:  Single dict, list of dicts, or DataFrame of cleaned-style rows.
        pipeline: Fitted sklearn Pipeline. If None, loads from default artifact path.

    Returns:
        List of InferenceResult named-tuples.
    """
    if pipeline is None:
        pipeline = load_pipeline()

    df = _coerce_to_frame(records)

    # Extract IDs for output (if present)
    customer_ids: list[str | None]
    if ID_COL in df.columns:
        customer_ids = df[ID_COL].tolist()
    else:
        customer_ids = [None] * len(df)

    # Drop target if accidentally included
    df = df.drop(columns=[TARGET_COL], errors="ignore")

    # Fill any missing columns with defaults
    df = _fill_missing_cols(df)

    # Keep only feature columns (ColumnTransformer will ignore extras via remainder="drop")
    X = df[ALL_FEATURE_COLS + [ID_COL] if ID_COL in df.columns else ALL_FEATURE_COLS]

    predictions: list[int] = pipeline.predict(X).tolist()
    probabilities: list[float] = pipeline.predict_proba(X)[:, 1].tolist()

    return [
        InferenceResult(
            customer_id=cid,
            prediction=pred,
            label="Yes" if pred == 1 else "No",
            churn_probability=round(prob, 4),
        )
        for cid, pred, prob in zip(customer_ids, predictions, probabilities)
    ]


def predict_proba(
    records: Any,
    pipeline: Pipeline | None = None,
) -> list[float]:
    """Return churn probabilities only (convenience wrapper around predict)."""
    return [r.churn_probability for r in predict(records, pipeline=pipeline)]
