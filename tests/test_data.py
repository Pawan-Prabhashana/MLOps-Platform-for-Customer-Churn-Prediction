"""Tests for the data ingestion, cleaning, and splitting pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from churnops.data.clean import clean_telco
from churnops.data.ingest import load_raw
from churnops.data.schema import TARGET_COL
from churnops.data.split import make_splits
from churnops.data.validate import validate

_REPO_ROOT = Path(__file__).parent.parent
_RAW_CSV = _REPO_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    return load_raw(_RAW_CSV)


@pytest.fixture(scope="module")
def clean_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    return clean_telco(raw_df)


@pytest.fixture(scope="module")
def splits(clean_df: pd.DataFrame):
    return make_splits(clean_df, seed=42)


# ── Cleaning tests ────────────────────────────────────────────────────────────

class TestCleaning:
    def test_total_charges_is_numeric(self, clean_df):
        assert pd.api.types.is_numeric_dtype(clean_df["TotalCharges"])

    def test_total_charges_no_nans(self, clean_df):
        assert clean_df["TotalCharges"].isna().sum() == 0

    def test_no_nans_anywhere(self, clean_df):
        nan_total = clean_df.isna().sum().sum()
        assert nan_total == 0, f"Found {nan_total} NaN values after cleaning"

    def test_target_is_binary(self, clean_df):
        vals = set(clean_df[TARGET_COL].unique())
        assert vals == {0, 1}, f"Target values should be {{0,1}}, got {vals}"

    def test_target_dtype_is_int(self, clean_df):
        assert clean_df[TARGET_COL].dtype == "int64"

    def test_row_count_preserved(self, raw_df, clean_df):
        assert len(clean_df) == len(raw_df)

    def test_customer_id_present(self, clean_df):
        assert "customerID" in clean_df.columns

    def test_raw_churn_column_removed(self, clean_df):
        assert "Churn" not in clean_df.columns

    def test_tenure_is_float(self, clean_df):
        assert clean_df["tenure"].dtype == "float64"

    def test_monthly_charges_is_float(self, clean_df):
        assert clean_df["MonthlyCharges"].dtype == "float64"


# ── Validation tests ──────────────────────────────────────────────────────────

class TestValidation:
    def test_validate_passes_on_clean_df(self, clean_df):
        validate(clean_df)  # should not raise

    def test_validate_raises_on_nan(self, clean_df):
        bad = clean_df.copy()
        bad.loc[0, "tenure"] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            validate(bad)

    def test_validate_raises_on_bad_target(self, clean_df):
        bad = clean_df.copy()
        bad.loc[0, TARGET_COL] = 99
        with pytest.raises(ValueError, match="unexpected values"):
            validate(bad)


# ── Split tests ───────────────────────────────────────────────────────────────

class TestSplits:
    def test_split_sizes_sum_to_original(self, clean_df, splits):
        total = len(splits.train) + len(splits.val) + len(splits.test)
        assert total == len(clean_df)

    def test_train_is_largest(self, splits):
        assert len(splits.train) > len(splits.val)
        assert len(splits.train) > len(splits.test)

    def test_val_and_test_roughly_equal(self, splits):
        ratio = len(splits.val) / len(splits.test)
        assert 0.8 <= ratio <= 1.2, f"val/test size ratio {ratio:.2f} out of expected range"

    def test_stratification_train(self, clean_df, splits):
        overall_rate = clean_df[TARGET_COL].mean()
        train_rate = splits.train[TARGET_COL].mean()
        assert abs(train_rate - overall_rate) < 0.02, (
            f"Train churn rate {train_rate:.3f} too far from overall {overall_rate:.3f}"
        )

    def test_stratification_val(self, clean_df, splits):
        overall_rate = clean_df[TARGET_COL].mean()
        val_rate = splits.val[TARGET_COL].mean()
        assert abs(val_rate - overall_rate) < 0.03

    def test_stratification_test(self, clean_df, splits):
        overall_rate = clean_df[TARGET_COL].mean()
        test_rate = splits.test[TARGET_COL].mean()
        assert abs(test_rate - overall_rate) < 0.03

    def test_reproducibility(self, clean_df):
        s1 = make_splits(clean_df, seed=42)
        s2 = make_splits(clean_df, seed=42)
        pd.testing.assert_frame_equal(s1.train, s2.train)
        pd.testing.assert_frame_equal(s1.val, s2.val)
        pd.testing.assert_frame_equal(s1.test, s2.test)

    def test_different_seeds_differ(self, clean_df):
        s1 = make_splits(clean_df, seed=42)
        s2 = make_splits(clean_df, seed=99)
        assert not s1.train.equals(s2.train)

    def test_no_overlap_between_splits(self, splits):
        train_ids = set(splits.train["customerID"])
        val_ids = set(splits.val["customerID"])
        test_ids = set(splits.test["customerID"])
        assert train_ids.isdisjoint(val_ids), "Train/val overlap"
        assert train_ids.isdisjoint(test_ids), "Train/test overlap"
        assert val_ids.isdisjoint(test_ids), "Val/test overlap"
