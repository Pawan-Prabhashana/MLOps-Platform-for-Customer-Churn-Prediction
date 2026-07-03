"""Stateless cleaning and type-casting for the raw Telco churn DataFrame.

Design contract
---------------
- Receives the raw DataFrame as loaded by ingest.load_raw().
- Returns a cleaned, correctly-typed DataFrame.
- Does NOT fit any encoders or scalers (that belongs in the model Pipeline,
  fit on train only, to prevent leakage).
- The returned DataFrame has zero NaN values.
"""

from __future__ import annotations

import logging

import pandas as pd

from churnops.data.schema import (
    BINARY_YN_COLS,
    NUMERIC_COLS,
    TARGET_COL,
    TARGET_RAW,
)

logger = logging.getLogger(__name__)

_YES_NO_MAP: dict[str, int] = {"Yes": 1, "No": 0}


def clean_telco(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps and return the cleaned DataFrame."""
    df = df.copy()

    # ── 1. Strip whitespace from every string column ──────────────────────────
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())
    logger.debug("Stripped whitespace from %d string columns.", len(str_cols))

    # ── 2. Fix TotalCharges (arrives as string; blank = brand-new customer) ───
    raw_total = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_blank = raw_total.isna().sum()
    if n_blank > 0:
        logger.info(
            "TotalCharges: %d blank/non-numeric rows found (all tenure=0). "
            "Filling with 0.0.",
            n_blank,
        )
    df["TotalCharges"] = raw_total.fillna(0.0)

    # ── 3. Cast all numeric columns to float64 ────────────────────────────────
    for col in NUMERIC_COLS:
        df[col] = df[col].astype("float64")

    # ── 4. Encode Yes/No string columns to int64 (1/0) ───────────────────────
    for col in BINARY_YN_COLS:
        df[col] = df[col].map(_YES_NO_MAP).astype("int64")

    # ── 5. Encode target: Churn Yes->1, No->0; rename to TARGET_COL ──────────
    df[TARGET_COL] = df[TARGET_RAW].map(_YES_NO_MAP).astype("int64")
    df = df.drop(columns=[TARGET_RAW])

    # ── 6. SeniorCitizen is already 0/1 int — ensure int64 ───────────────────
    df["SeniorCitizen"] = df["SeniorCitizen"].astype("int64")

    # ── 7. Final NaN guard ────────────────────────────────────────────────────
    n_nan = df.isna().sum().sum()
    if n_nan > 0:
        nan_cols = df.columns[df.isna().any()].tolist()
        raise ValueError(
            f"Cleaning left {n_nan} NaN values in columns: {nan_cols}"
        )

    logger.info(
        "Cleaning complete: %d rows × %d cols, 0 NaNs.", len(df), df.shape[1]
    )
    return df
