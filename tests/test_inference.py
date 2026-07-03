"""Tests for the inference layer — predict() on various input shapes and edge cases."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def fitted_pipeline():
    from churnops.models.train import train
    pipe, _ = train()
    return pipe


@pytest.fixture(scope="module")
def sample_row() -> dict:
    """One representative real-ish raw cleaned row (no target)."""
    return {
        "customerID": "TEST-0001",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": 1,
        "Dependents": 0,
        "tenure": 12.0,
        "PhoneService": 1,
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": 1,
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 53.85,
        "TotalCharges": 646.2,
    }


@pytest.fixture(scope="module")
def sample_rows(sample_row) -> list[dict]:
    row2 = {
        **sample_row,
        "customerID": "TEST-0002",
        "tenure": 60.0,
        "Contract": "Two year",
        "TotalCharges": 5000.0,
        "MonthlyCharges": 85.0,
    }
    row3 = {
        **sample_row,
        "customerID": "TEST-0003",
        "tenure": 1.0,
        "TotalCharges": 0.0,
        "Contract": "Month-to-month",
    }
    return [sample_row, row2, row3]


# ── Input-type handling ───────────────────────────────────────────────────────

class TestInputTypes:
    def test_predict_single_dict(self, sample_row, fitted_pipeline):
        from churnops.models.inference import predict
        results = predict(sample_row, pipeline=fitted_pipeline)
        assert len(results) == 1

    def test_predict_list_of_dicts(self, sample_rows, fitted_pipeline):
        from churnops.models.inference import predict
        results = predict(sample_rows, pipeline=fitted_pipeline)
        assert len(results) == 3

    def test_predict_dataframe(self, sample_rows, fitted_pipeline):
        from churnops.models.inference import predict
        df = pd.DataFrame(sample_rows)
        results = predict(df, pipeline=fitted_pipeline)
        assert len(results) == len(df)

    def test_predict_with_target_column_included(self, sample_row, fitted_pipeline):
        """target column in input must be silently dropped, not crash."""
        from churnops.models.inference import predict
        row_with_target = {**sample_row, "churn": 1}
        results = predict(row_with_target, pipeline=fitted_pipeline)
        assert len(results) == 1


# ── Output shape and types ────────────────────────────────────────────────────

class TestOutputFormat:
    def test_prediction_is_0_or_1(self, sample_rows, fitted_pipeline):
        from churnops.models.inference import predict
        for r in predict(sample_rows, pipeline=fitted_pipeline):
            assert r.prediction in {0, 1}

    def test_label_is_yes_or_no(self, sample_rows, fitted_pipeline):
        from churnops.models.inference import predict
        for r in predict(sample_rows, pipeline=fitted_pipeline):
            assert r.label in {"Yes", "No"}

    def test_probability_in_unit_interval(self, sample_rows, fitted_pipeline):
        from churnops.models.inference import predict
        for r in predict(sample_rows, pipeline=fitted_pipeline):
            assert 0.0 <= r.churn_probability <= 1.0

    def test_customer_id_returned(self, sample_row, fitted_pipeline):
        from churnops.models.inference import predict
        results = predict(sample_row, pipeline=fitted_pipeline)
        assert results[0].customer_id == sample_row["customerID"]

    def test_predict_proba_returns_floats(self, sample_rows, fitted_pipeline):
        from churnops.models.inference import predict_proba
        probs = predict_proba(sample_rows, pipeline=fitted_pipeline)
        assert len(probs) == 3
        assert all(isinstance(p, float) for p in probs)


# ── Robustness ────────────────────────────────────────────────────────────────

class TestRobustness:
    def test_unseen_categorical_value_no_crash(self, sample_row, fitted_pipeline):
        """handle_unknown='ignore' in OHE means unknown categories → all-zero columns."""
        from churnops.models.inference import predict
        row = {**sample_row, "Contract": "UNSEEN_CONTRACT_TYPE"}
        results = predict(row, pipeline=fitted_pipeline)
        assert len(results) == 1
        assert results[0].prediction in {0, 1}

    def test_unseen_gender_no_crash(self, sample_row, fitted_pipeline):
        from churnops.models.inference import predict
        row = {**sample_row, "gender": "Unknown"}
        results = predict(row, pipeline=fitted_pipeline)
        assert results[0].prediction in {0, 1}

    def test_missing_optional_column_fills_default(self, sample_row, fitted_pipeline):
        """A row missing a non-critical column should still return a prediction."""
        from churnops.models.inference import predict
        row = {k: v for k, v in sample_row.items() if k != "OnlineSecurity"}
        results = predict(row, pipeline=fitted_pipeline)
        assert results[0].prediction in {0, 1}

    def test_high_tenure_low_churn_probability(self, fitted_pipeline):
        """Sanity: a 6-year customer on a two-year contract should have low churn probability."""
        from churnops.models.inference import predict
        row = {
            "customerID": "SANITY-001",
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": 1,
            "Dependents": 1,
            "tenure": 72.0,
            "PhoneService": 1,
            "MultipleLines": "Yes",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Two year",
            "PaperlessBilling": 0,
            "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": 110.0,
            "TotalCharges": 7920.0,
        }
        results = predict(row, pipeline=fitted_pipeline)
        assert results[0].churn_probability < 0.5, (
            f"Expected low churn for loyal long-term customer, got {results[0].churn_probability}"
        )
