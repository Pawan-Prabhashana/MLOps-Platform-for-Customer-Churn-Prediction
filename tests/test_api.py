"""Tests for the FastAPI prediction API.

Uses FastAPI's TestClient and dependency-override to inject a pre-loaded model
state, so no MLflow server or Docker is required. Tests run fast and in-process.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

_REPO_ROOT = Path(__file__).parent.parent

# ── Fixtures & helpers ────────────────────────────────────────────────────────

_VALID_RECORD = {
    "customerID": "7590-VHVEG",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}


def _joblib_model() -> Any:
    """Load the local joblib pipeline — fast (~7 KB) and registry-free."""
    from churnops.models.persistence import load_pipeline
    return load_pipeline()


def _make_client(model: Any | None = None, *, loaded: bool = True) -> TestClient:
    """Return a TestClient with the model state dependency overridden."""
    from churnops.api.app import create_app
    from churnops.api.deps import get_model_state

    pipeline = model or _joblib_model()

    def _override_model_state():
        if not loaded:
            from fastapi import HTTPException, status
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded (test stub).",
            )
        return {
            "model_loaded": True,
            "model": pipeline,
            "source": "joblib:test-stub",
            "version": "test-1",
            "load_time": "2025-01-01T00:00:00+00:00",
            "startup_ts": 0.0,
            "uptime_seconds": 1.0,
        }

    app = create_app()

    # Patch app.state so /health also works (middleware reads state directly).
    app.state.model_state = _override_model_state() if loaded else {
        "model_loaded": False,
        "model": None,
        "source": None,
        "version": None,
        "load_time": None,
        "startup_ts": 0.0,
    }
    app.dependency_overrides[get_model_state] = _override_model_state

    # Skip the lifespan (we set state directly above).
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return _make_client()


# ── /health ───────────────────────────────────────────────────────────────────


class TestHealth:
    def test_health_ok_when_model_loaded(self, client: TestClient) -> None:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True

    def test_health_503_when_model_not_loaded(self) -> None:
        degraded_client = _make_client(loaded=False)
        r = degraded_client.get("/health")
        assert r.status_code == 503


# ── /model-info ───────────────────────────────────────────────────────────────


class TestModelInfo:
    def test_model_info_returns_schema_fields(self, client: TestClient) -> None:
        r = client.get("/model-info")
        assert r.status_code == 200
        body = r.json()
        assert "feature_columns" in body
        assert len(body["feature_columns"]) > 0
        assert "threshold" in body
        assert body["threshold"] == pytest.approx(0.5)


# ── POST /predict ─────────────────────────────────────────────────────────────


class TestPredict:
    def test_valid_record_returns_prediction(self, client: TestClient) -> None:
        r = client.post("/predict", json=_VALID_RECORD)
        assert r.status_code == 200
        body = r.json()
        assert 0.0 <= body["churn_probability"] <= 1.0
        assert body["prediction"] in {"Yes", "No"}
        assert body["customerID"] == "7590-VHVEG"

    def test_threshold_override_flips_label(self, client: TestClient) -> None:
        # Get the actual probability first, then bracket it from both sides.
        base = client.post("/predict", json=_VALID_RECORD).json()
        prob = base["churn_probability"]

        above = client.post(f"/predict?threshold={max(0.0, prob - 0.01)}", json=_VALID_RECORD).json()
        below = client.post(f"/predict?threshold={min(1.0, prob + 0.01)}", json=_VALID_RECORD).json()

        assert above["prediction"] == "Yes"
        assert below["prediction"] == "No"

    def test_unseen_categorical_does_not_500(self, client: TestClient) -> None:
        record = dict(_VALID_RECORD)
        record["Contract"] = "Quantum-entangled 999yr plan"
        r = client.post("/predict", json=record)
        # Must be 200 — pipeline has handle_unknown="ignore"
        assert r.status_code == 200
        assert 0.0 <= r.json()["churn_probability"] <= 1.0

    def test_missing_required_field_returns_422(self, client: TestClient) -> None:
        bad = dict(_VALID_RECORD)
        del bad["tenure"]
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_wrong_type_returns_422(self, client: TestClient) -> None:
        bad = dict(_VALID_RECORD, tenure="not-a-number")
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_extra_fields_ignored_not_500(self, client: TestClient) -> None:
        """Extra fields (e.g. ground-truth Churn) must be silently ignored."""
        with_churn = dict(_VALID_RECORD, Churn="No")
        r = client.post("/predict", json=with_churn)
        assert r.status_code == 200

    def test_503_when_model_not_loaded(self) -> None:
        degraded = _make_client(loaded=False)
        r = degraded.post("/predict", json=_VALID_RECORD)
        assert r.status_code == 503


# ── POST /predict/batch ───────────────────────────────────────────────────────


class TestPredictBatch:
    def test_batch_returns_one_prediction_per_record(self, client: TestClient) -> None:
        body = {"records": [_VALID_RECORD, _VALID_RECORD]}
        r = client.post("/predict/batch", json=body)
        assert r.status_code == 200
        resp = r.json()
        assert resp["count"] == 2
        assert len(resp["predictions"]) == 2

    def test_batch_summary_churn_rate_correct(self, client: TestClient) -> None:
        # Two identical records → churn_rate must equal 0.0 or 1.0 (consistent).
        body = {"records": [_VALID_RECORD, _VALID_RECORD]}
        r = client.post("/predict/batch", json=body).json()
        preds = r["predictions"]
        n_yes = sum(1 for p in preds if p["prediction"] == "Yes")
        expected_rate = round(n_yes / len(preds), 4)
        assert r["summary"]["churn_rate"] == pytest.approx(expected_rate)

    def test_batch_threshold_override(self, client: TestClient) -> None:
        # threshold=0 → all "Yes", threshold=1 → all "No"
        body_all_yes = {"records": [_VALID_RECORD], "threshold": 0.0}
        body_all_no  = {"records": [_VALID_RECORD], "threshold": 1.0}
        assert client.post("/predict/batch", json=body_all_yes).json()["predictions"][0]["prediction"] == "Yes"
        assert client.post("/predict/batch", json=body_all_no).json()["predictions"][0]["prediction"] == "No"

    def test_oversized_batch_rejected_413(self, client: TestClient) -> None:
        # configs/api.yaml sets max_records=500; send 501.
        big_batch = {"records": [_VALID_RECORD] * 501}
        r = client.post("/predict/batch", json=big_batch)
        assert r.status_code == 413

    def test_empty_batch_returns_422(self, client: TestClient) -> None:
        r = client.post("/predict/batch", json={"records": []})
        assert r.status_code == 422
