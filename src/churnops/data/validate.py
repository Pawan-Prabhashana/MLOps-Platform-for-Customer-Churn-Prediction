"""Schema and quality validation for the cleaned Telco churn DataFrame.

Raises ValueError on the first violation found.
"""

from __future__ import annotations

import logging

import pandas as pd

from churnops.data.schema import ALL_FEATURE_COLS, EXPECTED_DTYPES, ID_COL, TARGET_COL

logger = logging.getLogger(__name__)


def validate(df: pd.DataFrame) -> None:
    """Validate a cleaned DataFrame.  Raises ValueError if any check fails."""

    errors: list[str] = []

    # ── 1. Required columns present ───────────────────────────────────────────
    required = {ID_COL, TARGET_COL, *ALL_FEATURE_COLS}
    missing = required - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {sorted(missing)}")

    # ── 2. No NaN values anywhere ─────────────────────────────────────────────
    nan_counts = df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if not nan_cols.empty:
        errors.append(f"NaN values found: {nan_cols.to_dict()}")

    # ── 3. Target is strictly {0, 1} ──────────────────────────────────────────
    if TARGET_COL in df.columns:
        bad_vals = set(df[TARGET_COL].unique()) - {0, 1}
        if bad_vals:
            errors.append(f"Target '{TARGET_COL}' contains unexpected values: {bad_vals}")

    # ── 4. Dtype conformance ──────────────────────────────────────────────────
    for col, expected in EXPECTED_DTYPES.items():
        if col not in df.columns:
            continue
        actual = str(df[col].dtype)
        if actual != expected:
            errors.append(
                f"Column '{col}': expected dtype '{expected}', got '{actual}'"
            )

    # ── 5. No duplicate customerIDs ───────────────────────────────────────────
    if ID_COL in df.columns:
        n_dupes = df[ID_COL].duplicated().sum()
        if n_dupes > 0:
            errors.append(f"{n_dupes} duplicate '{ID_COL}' values found")

    # ── 6. Row count sanity ───────────────────────────────────────────────────
    if len(df) < 100:
        errors.append(f"Suspiciously few rows: {len(df)}")

    if errors:
        raise ValueError("Validation failed:\n  - " + "\n  - ".join(errors))

    logger.info("Validation passed: %d rows, %d columns.", len(df), df.shape[1])
