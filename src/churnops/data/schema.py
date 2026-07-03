"""Column schema for the Telco churn dataset.

Single source of truth for column groupings.  Reads from configs/data.yaml
so that schema.py and data.yaml are never out of sync — edit the YAML, not
the Python constants below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import yaml

_REPO_ROOT: Final[Path] = Path(__file__).parent.parent.parent.parent
_DATA_YAML: Final[Path] = _REPO_ROOT / "configs" / "data.yaml"


def _load() -> dict:
    with _DATA_YAML.open() as f:
        return yaml.safe_load(f)["schema"]


_s = _load()

# ── Identifier ────────────────────────────────────────────────────────────────
ID_COL: Final[str] = _s["id_col"]

# ── Target ────────────────────────────────────────────────────────────────────
TARGET_RAW: Final[str] = _s["target_raw"]        # "Churn"  (Yes/No string)
TARGET_COL: Final[str] = _s["target_col"]        # "churn"  (0/1 integer)

# ── Feature groups ────────────────────────────────────────────────────────────
NUMERIC_COLS: Final[list[str]] = _s["numeric_cols"]
INTEGER_BINARY_COLS: Final[list[str]] = _s["integer_binary_cols"]
BINARY_YN_COLS: Final[list[str]] = _s["binary_yn_cols"]
CATEGORICAL_COLS: Final[list[str]] = _s["categorical_cols"]

# ── Convenience aggregates ────────────────────────────────────────────────────
ALL_FEATURE_COLS: Final[list[str]] = (
    NUMERIC_COLS + INTEGER_BINARY_COLS + BINARY_YN_COLS + CATEGORICAL_COLS
)

ALL_COLS: Final[list[str]] = [ID_COL] + ALL_FEATURE_COLS + [TARGET_COL]

# ── Expected dtypes after cleaning (for validation) ───────────────────────────
EXPECTED_DTYPES: Final[dict[str, str]] = {
    **{col: "float64" for col in NUMERIC_COLS},
    **{col: "int64" for col in INTEGER_BINARY_COLS},
    **{col: "int64" for col in BINARY_YN_COLS},
    **{col: "object" for col in CATEGORICAL_COLS},
    TARGET_COL: "int64",
}
