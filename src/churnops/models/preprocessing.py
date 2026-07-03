"""Build the ColumnTransformer preprocessor from the schema column lists.

Design
------
- numeric columns    → StandardScaler
- integer binary cols (already 0/1) → passthrough
- Yes/No binary cols (already 0/1 after cleaning) → passthrough
- multi-class categorical cols → OneHotEncoder(handle_unknown="ignore")
  so unseen categories at inference time are represented as all-zeros,
  not an error.
- customerID is dropped (remainder="drop") — never a feature.

Returns an *unfitted* ColumnTransformer; the Pipeline fits it on train only.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churnops.data.schema import (
    BINARY_YN_COLS,
    CATEGORICAL_COLS,
    INTEGER_BINARY_COLS,
    NUMERIC_COLS,
)

# All binary columns (already 0/1 — no encoding needed, just pass through)
_BINARY_COLS = INTEGER_BINARY_COLS + BINARY_YN_COLS


def build_preprocessor() -> ColumnTransformer:
    """Return an unfitted ColumnTransformer ready to be embedded in a Pipeline."""
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_COLS,
            ),
            (
                "binary",
                "passthrough",
                _BINARY_COLS,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=float,
                ),
                CATEGORICAL_COLS,
            ),
        ],
        remainder="drop",  # drops customerID and any unexpected columns
        verbose_feature_names_out=False,
    )
