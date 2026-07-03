"""Stratified train / val / test splitting.

Design
------
Two-step sklearn StratifiedShuffleSplit:
  1. Split full data → train  (70%) + temp (30%)
  2. Split temp      → val    (15%) + test (15%)

The seed is applied consistently so that re-running with the same seed
always produces identical splits.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from churnops.data.schema import TARGET_COL

logger = logging.getLogger(__name__)


class DataSplits(NamedTuple):
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def make_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> DataSplits:
    """Return stratified (train, val, test) splits.

    Args:
        df: Cleaned DataFrame containing TARGET_COL.
        train_ratio: Fraction of data for training set.
        val_ratio: Fraction of data for validation set.
        test_ratio: Fraction of data for test set.
        seed: Random seed for full reproducibility.

    Returns:
        DataSplits named-tuple with .train, .val, .test DataFrames.
    """
    total = train_ratio + val_ratio + test_ratio
    if not abs(total - 1.0) < 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")

    y = df[TARGET_COL]

    # ── Step 1: train vs temp ─────────────────────────────────────────────────
    temp_ratio = 1.0 - train_ratio
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=temp_ratio, random_state=seed)
    train_idx, temp_idx = next(sss1.split(df, y))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    # ── Step 2: val vs test from temp ─────────────────────────────────────────
    val_fraction_of_temp = val_ratio / temp_ratio
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=1.0 - val_fraction_of_temp, random_state=seed
    )
    val_idx, test_idx = next(sss2.split(temp_df, temp_df[TARGET_COL]))

    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    _log_split_stats("train", train_df)
    _log_split_stats("val", val_df)
    _log_split_stats("test", test_df)

    assert len(train_df) + len(val_df) + len(test_df) == len(df), (
        "Split row counts don't add up to original length"
    )

    return DataSplits(train=train_df, val=val_df, test=test_df)


def _log_split_stats(name: str, df: pd.DataFrame) -> None:
    churn_rate = df[TARGET_COL].mean() * 100
    logger.info("  %-6s  %5d rows  churn=%.1f%%", name, len(df), churn_rate)
