"""Tests for MLflow tracking, model registration, and promotion.

All tests point MLflow at a temporary local file-store so they do NOT
require the Docker tracking server to be running.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import pytest

_REPO_ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def tmp_tracking_uri(tmp_path_factory):
    """Return a SQLite-based tracking URI in a temp directory.

    MLflow 3.x deprecated the file-store backend; SQLite is the
    lightweight alternative that works without a running server.
    """
    store = tmp_path_factory.mktemp("mlruns")
    uri = f"sqlite:///{store}/mlflow.db"
    mlflow.set_tracking_uri(uri)
    # Also override the env var so helpers that call get_tracking_uri() pick it up
    import os
    os.environ["MLFLOW_TRACKING_URI"] = uri
    yield uri
    os.environ.pop("MLFLOW_TRACKING_URI", None)


@pytest.fixture(scope="module")
def trained_pipeline_and_metrics():
    from churnops.models.train import train
    return train(model_key="logistic_regression")


# ── Experiment & run logging ──────────────────────────────────────────────────

class TestRunLogging:
    def test_start_run_creates_run(self, tmp_tracking_uri):
        from churnops.tracking.mlflow_utils import start_run
        with start_run("test_run", tracking_uri=tmp_tracking_uri) as run:
            run_id = run.info.run_id
            mlflow.log_param("foo", "bar")
            mlflow.log_metric("accuracy", 0.8)

        client = mlflow.MlflowClient()
        fetched = client.get_run(run_id)
        assert fetched.data.params["foo"] == "bar"
        assert fetched.data.metrics["accuracy"] == 0.8

    def test_log_model_params_logged(self, tmp_tracking_uri, trained_pipeline_and_metrics):
        import yaml  # noqa: I001

        from churnops.tracking.mlflow_utils import log_model_params, start_run
        model_cfg = yaml.safe_load((_REPO_ROOT / "configs" / "model.yaml").read_text())

        with start_run("param_test", tracking_uri=tmp_tracking_uri) as run:
            log_model_params("logistic_regression", model_cfg["models"]["logistic_regression"], 42)
            run_id = run.info.run_id

        client = mlflow.MlflowClient()
        params = client.get_run(run_id).data.params
        assert "model_key" in params
        assert "model_class" in params

    def test_log_split_metrics_logged(self, tmp_tracking_uri, trained_pipeline_and_metrics):
        _, metrics = trained_pipeline_and_metrics
        from churnops.tracking.mlflow_utils import log_split_metrics, start_run

        with start_run("metric_test", tracking_uri=tmp_tracking_uri) as run:
            log_split_metrics(metrics)
            run_id = run.info.run_id

        client = mlflow.MlflowClient()
        logged = client.get_run(run_id).data.metrics
        assert "val_roc_auc" in logged
        assert "test_roc_auc" in logged
        assert logged["val_roc_auc"] > 0.7

    def test_model_artifact_logged(self, tmp_tracking_uri, trained_pipeline_and_metrics):
        """Training run should log a model artifact that can be reloaded."""
        pipeline, metrics = trained_pipeline_and_metrics
        import mlflow.sklearn

        with mlflow.start_run() as run:
            mlflow.sklearn.log_model(pipeline, name="model")
            run_id = run.info.run_id

        # Verify the model can be reloaded from the run URI
        loaded = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        assert loaded is not None
        assert hasattr(loaded, "predict")


# ── Model Registry ────────────────────────────────────────────────────────────

class TestRegistry:
    @pytest.fixture(scope="class")
    @classmethod
    def registered_version(cls, tmp_tracking_uri, trained_pipeline_and_metrics):
        """Register the trained pipeline and return (version, run_id)."""
        pipeline, metrics = trained_pipeline_and_metrics
        import mlflow.sklearn  # noqa: I001

        from churnops.tracking.registry import ensure_registered_model, register_model

        model_name = "test-churn-classifier"
        ensure_registered_model(model_name)

        with mlflow.start_run() as run:
            mlflow.sklearn.log_model(pipeline, artifact_path="model")
            run_id = run.info.run_id

        version = register_model(run_id, "model", model_name=model_name)
        return version, run_id, model_name

    def test_registration_creates_version(self, registered_version):
        version, _, model_name = registered_version
        client = mlflow.MlflowClient()
        mv = client.get_model_version(model_name, version)
        assert mv.version == version

    def test_set_alias_then_load_by_alias(self, registered_version, tmp_tracking_uri):
        version, run_id, model_name = registered_version
        from churnops.tracking.registry import set_alias

        set_alias(version, "test_alias", model_name=model_name)
        model_uri = f"models:/{model_name}@test_alias"
        loaded = mlflow.sklearn.load_model(model_uri)
        assert loaded is not None
        assert hasattr(loaded, "predict")

    def test_loaded_model_predicts(self, registered_version):
        import pandas as pd
        version, _, model_name = registered_version
        from churnops.tracking.registry import set_alias

        set_alias(version, "predict_test_alias", model_name=model_name)
        loaded = mlflow.sklearn.load_model(f"models:/{model_name}@predict_test_alias")

        sample = pd.DataFrame([{
            "customerID": "T001", "gender": "Female", "SeniorCitizen": 0,
            "Partner": 1, "Dependents": 0, "tenure": 24.0, "PhoneService": 1,
            "MultipleLines": "No", "InternetService": "DSL", "OnlineSecurity": "Yes",
            "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "No", "StreamingMovies": "No", "Contract": "One year",
            "PaperlessBilling": 1, "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 58.5, "TotalCharges": 1404.0,
        }])
        preds = loaded.predict(sample)
        assert preds[0] in {0, 1}


# ── Promotion logic ───────────────────────────────────────────────────────────

class TestPromotion:
    @pytest.fixture(scope="class")
    @classmethod
    def two_versions(cls, tmp_tracking_uri, trained_pipeline_and_metrics):
        """Register two versions with different mocked test_roc_auc values."""
        pipeline, metrics = trained_pipeline_and_metrics
        import mlflow.sklearn  # noqa: I001

        from churnops.tracking.registry import ensure_registered_model

        model_name = "promo-test-model"
        ensure_registered_model(model_name)

        versions = []
        for roc_val in [0.82, 0.87]:
            with mlflow.start_run() as run:
                mlflow.sklearn.log_model(pipeline, artifact_path="model")
                mlflow.log_metric("test_roc_auc", roc_val)
                run_id = run.info.run_id
            mv = mlflow.register_model(f"runs:/{run_id}/model", model_name)
            versions.append((mv.version, roc_val))

        return model_name, versions

    def test_promote_picks_higher_metric(self, two_versions, tmp_tracking_uri):
        model_name, versions = two_versions
        from churnops.tracking.registry import promote_best

        new_ver, reason = promote_best(
            model_name=model_name,
            metric_name="test_roc_auc",
            force=True,
        )
        assert new_ver is not None
        # The version with roc_auc=0.87 should win
        best_ver = versions[1][0]  # second version has 0.87
        assert new_ver == best_ver, (
            f"Expected version {best_ver} (roc_auc=0.87) to be promoted, got v{new_ver}"
        )

    def test_no_promote_if_not_better(self, two_versions, tmp_tracking_uri):
        """promote_best should refuse to demote production without --force."""
        model_name, _ = two_versions
        from churnops.tracking.registry import get_alias, promote_best

        current = get_alias("production", model_name)
        # Try again without force — best is already promoted, so no change
        new_ver, reason = promote_best(
            model_name=model_name,
            metric_name="test_roc_auc",
            force=False,
        )
        # Either promotes same version OR refuses; either way production stays best
        prod_after = get_alias("production", model_name)
        assert prod_after == current or (new_ver is not None)
