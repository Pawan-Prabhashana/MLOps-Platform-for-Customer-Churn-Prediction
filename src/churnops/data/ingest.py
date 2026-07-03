"""Raw CSV loader for the Telco churn dataset."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load the raw Telco CSV and return a DataFrame with original dtypes."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {path}")

    df = pd.read_csv(path)
    logger.info("Loaded raw CSV: %s rows × %s cols from %s", len(df), df.shape[1], path)
    return df
