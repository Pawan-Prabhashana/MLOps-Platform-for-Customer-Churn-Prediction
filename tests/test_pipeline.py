"""Tests for the sklearn Pipeline: training, metrics, reproducibility, persistence."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

_REPO_ROOT = Path(__file__).parent.parent

# Load metric thresholds from config once
with (_REPO_ROOT / "configs" / "model.yaml").open() as _f:
    _MODEL_CFG = yaml.safe_load(_f)
_THRESHOLDS = _MODEL_CFG["thresholds"]


@pytest.fixture(scope="module")
def pipeline_and_metrics():
    """Train the default model once and cache the result for the whole module."""
    from churnops.models.train import train
    return train()


@pytest.fixture(scope="module")
def fitted_pipeline(pipeline_and_metrics):
    return pipeline_and_metrics[0]


@pytest.fixture(scope="module")
def metrics(pipeline_and_metrics):
    return pipeline_and_metrics[1]


@pytest.fixture(scope="module")
def val_df():
    cfg = yaml.safe_load((_REPO_ROOT / "configs" / "data.yaml").read_text())
    return pd.read_parquet(
        _REPO_ROOT / cfg["paths"]["processed_dir"] / cfg["paths"]["val_file"]
    )


@pytest.fixture(scope="module")
def test_df():
    cfg = yaml.safe_load((_REPO_ROOT / "configs" / "data.yaml").read_text())
    return pd.read_parquet(
        _REPO_ROOT / cfg["paths"]["processed_dir"] / cfg["paths"]["test_file"]
    )


# ── Training & metrics ────────────────────────────────────────────────────────

class TestTraining:
    def test_pipeline_trains_without_error(self, fitted_pipeline):
        assert fitted_pipeline is not None

    def test_pipeline_has_two_steps(self, fitted_pipeline):
        assert list(fitted_pipeline.named_steps.keys()) == ["preprocess", "model"]

    def test_val_roc_auc_above_threshold(self, metrics):
        auc = metrics["val"]["roc_auc"]
        assert auc >= _THRESHOLDS["roc_auc"], (
            f"Val ROC-AUC {auc:.4f} below threshold {_THRESHOLDS['roc_auc']}"
        )

    def test_test_roc_auc_above_threshold(self, metrics):
        auc = metrics["test"]["roc_auc"]
        assert auc >= _THRESHOLDS["roc_auc"], (
            f"Test ROC-AUC {auc:.4f} below threshold {_THRESHOLDS['roc_auc']}"
        )

    def test_val_pr_auc_above_threshold(self, metrics):
        assert metrics["val"]["pr_auc"] >= _THRESHOLDS["pr_auc"]

    def test_val_recall_above_threshold(self, metrics):
        assert metrics["val"]["recall"] >= _THRESHOLDS["recall"]

    def test_predictions_are_binary(self, fitted_pipeline, val_df):
        from churnops.data.schema import TARGET_COL
        X = val_df.drop(columns=[TARGET_COL])
        preds = fitted_pipeline.predict(X)
        assert set(preds.tolist()).issubset({0, 1})

    def test_probabilities_in_unit_interval(self, fitted_pipeline, val_df):
        from churnops.data.schema import TARGET_COL
        X = val_df.drop(columns=[TARGET_COL])
        proba = fitted_pipeline.predict_proba(X)
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0
        assert proba.shape[1] == 2


# ── Reproducibility ───────────────────────────────────────────────────────────

class TestReproducibility:
    def test_same_seed_same_metrics(self):
        from churnops.models.train import train
        _, m1 = train()
        _, m2 = train()
        assert m1["val"]["roc_auc"] == m2["val"]["roc_auc"]
        assert m1["test"]["roc_auc"] == m2["test"]["roc_auc"]


# ── Persistence: save → reload → identical predictions ───────────────────────

class TestPersistence:
    def test_save_and_reload_identical_predictions(self, fitted_pipeline, tmp_path, val_df):
        from churnops.data.schema import TARGET_COL
        from churnops.models.persistence import load_pipeline, save_pipeline
        from churnops.models.train import train

        pipe_path = tmp_path / "pipeline.joblib"
        side_path = tmp_path / "pipeline_meta.json"
        _, metrics = train()
        save_pipeline(fitted_pipeline, metrics, pipeline_path=pipe_path, sidecar_path=side_path)

        reloaded = load_pipeline(pipe_path)
        X = val_df.drop(columns=[TARGET_COL])

        import numpy as np
        preds_orig = fitted_pipeline.predict(X)
        preds_relo = reloaded.predict(X)
        assert np.array_equal(preds_orig, preds_relo), "Reloaded pipeline gives different predictions"

    def test_sidecar_contains_required_keys(self, tmp_path):
        from churnops.models.persistence import save_pipeline
        from churnops.models.train import train

        pipe, metrics = train()
        pipe_path = tmp_path / "p.joblib"
        side_path = tmp_path / "meta.json"
        save_pipeline(pipe, metrics, pipeline_path=pipe_path, sidecar_path=side_path)

        import json
        sidecar = json.loads(side_path.read_text())
        for key in ("schema_version", "saved_at", "model_key", "val_metrics", "test_metrics"):
            assert key in sidecar, f"Sidecar missing key '{key}'"
