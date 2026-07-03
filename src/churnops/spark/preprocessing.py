"""MLlib feature-engineering stages built from the shared schema.

Mirrors churnops.models.preprocessing (the sklearn ColumnTransformer) so the two
paths are comparable:

    sklearn                          →  MLlib equivalent
    ------------------------------------------------------------------
    StandardScaler (numeric)         →  VectorAssembler + StandardScaler
    passthrough (0/1 binary cols)    →  used as-is in the final assembler
    OneHotEncoder(handle_unknown=    →  StringIndexer(handleInvalid="keep")
        "ignore") (categorical)          + OneHotEncoder
    drop customerID                  →  customerID never enters the assembler

Scaling approach (documented choice): we scale the numeric columns BEFORE the
final assembly. Numeric columns are assembled into their own vector, standardized
with StandardScaler, then concatenated with the (already 0/1) binary columns and
the one-hot encoded categoricals into the final "features" vector. This keeps the
binary/one-hot columns unscaled — exactly matching the sklearn design where only
the numeric block is passed through StandardScaler.

The churn target ("churn", already 0/1 in the parquet) is used directly as the
MLlib label column, so no label indexing is required.
"""

from __future__ import annotations

from pyspark.ml import Transformer
from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler

from churnops.data.schema import (
    BINARY_YN_COLS,
    CATEGORICAL_COLS,
    INTEGER_BINARY_COLS,
    NUMERIC_COLS,
)

# All binary columns are already 0/1 after cleaning — no encoding, straight into
# the final assembler.
_BINARY_COLS = INTEGER_BINARY_COLS + BINARY_YN_COLS

FEATURES_COL = "features"
NUMERIC_VEC_COL = "numeric_vec"
NUMERIC_SCALED_COL = "numeric_scaled"


def _indexed_col(col: str) -> str:
    return f"{col}_idx"


def _ohe_col(col: str) -> str:
    return f"{col}_ohe"


def build_feature_stages() -> list[Transformer]:
    """Return the ordered list of MLlib feature-engineering stages.

    Order:
      1. StringIndexer per categorical column (handleInvalid="keep").
      2. OneHotEncoder over all indexed categorical columns.
      3. VectorAssembler of numeric columns → numeric_vec.
      4. StandardScaler(numeric_vec) → numeric_scaled.
      5. VectorAssembler of [numeric_scaled] + binary + one-hot → "features".
    """
    stages: list[Transformer] = []

    # 1. Index each categorical column. handleInvalid="keep" maps unseen
    #    categories at inference time to a dedicated extra index instead of
    #    throwing — the equivalent of sklearn's handle_unknown="ignore".
    for col in CATEGORICAL_COLS:
        stages.append(
            StringIndexer(
                inputCol=col,
                outputCol=_indexed_col(col),
                handleInvalid="keep",
            )
        )

    # 2. One-hot encode all indexed categoricals in a single stage.
    if CATEGORICAL_COLS:
        stages.append(
            OneHotEncoder(
                inputCols=[_indexed_col(c) for c in CATEGORICAL_COLS],
                outputCols=[_ohe_col(c) for c in CATEGORICAL_COLS],
                handleInvalid="keep",
            )
        )

    # 3. Assemble numeric columns, then 4. standardize them (numeric block only).
    stages.append(
        VectorAssembler(
            inputCols=NUMERIC_COLS,
            outputCol=NUMERIC_VEC_COL,
            handleInvalid="keep",
        )
    )
    stages.append(
        StandardScaler(
            inputCol=NUMERIC_VEC_COL,
            outputCol=NUMERIC_SCALED_COL,
            withMean=True,
            withStd=True,
        )
    )

    # 5. Final assembler: scaled numerics + raw binaries + one-hot categoricals.
    final_inputs = [NUMERIC_SCALED_COL] + _BINARY_COLS + [_ohe_col(c) for c in CATEGORICAL_COLS]
    stages.append(
        VectorAssembler(
            inputCols=final_inputs,
            outputCol=FEATURES_COL,
            handleInvalid="keep",
        )
    )

    return stages
