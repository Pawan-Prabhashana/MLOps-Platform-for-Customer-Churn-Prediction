"""Tests for the central Settings object."""

from pathlib import Path

from churnops.config import Settings, get_settings, settings


class TestSettingsDefaults:
    """Verify that Settings loads with expected default values."""

    def test_settings_is_importable(self):
        assert settings is not None

    def test_get_settings_returns_settings_instance(self):
        s = get_settings()
        assert isinstance(s, Settings)

    def test_get_settings_is_cached(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    # ── Random seed ──────────────────────────────────────────────────────────
    def test_random_seed_default(self):
        assert settings.random_seed == 42

    # ── Kafka topics ─────────────────────────────────────────────────────────
    def test_kafka_bootstrap_servers(self):
        assert settings.kafka_bootstrap_servers == "localhost:9092"

    def test_kafka_topic_raw_customers(self):
        assert settings.kafka_topic_raw_customers == "telco.raw.customers"

    def test_kafka_topic_predictions(self):
        assert settings.kafka_topic_predictions == "telco.churn.predictions"

    def test_kafka_topic_deadletter(self):
        assert settings.kafka_topic_deadletter == "telco.deadletter"

    # ── MLflow ────────────────────────────────────────────────────────────────
    def test_mlflow_tracking_uri(self):
        assert "3000" in settings.mlflow_tracking_uri or "localhost" in settings.mlflow_tracking_uri

    # ── Paths exist as Path objects ───────────────────────────────────────────
    def test_raw_data_dir_is_path(self):
        assert isinstance(settings.raw_data_dir, Path)

    def test_processed_data_dir_is_path(self):
        assert isinstance(settings.processed_data_dir, Path)

    def test_artifacts_dir_is_path(self):
        assert isinstance(settings.artifacts_dir, Path)

    # ── Derived URIs contain expected components ──────────────────────────────
    def test_mlflow_db_uri_contains_db_name(self):
        assert "mlflow" in settings.mlflow_db_uri
        assert "postgresql" in settings.mlflow_db_uri

    def test_airflow_db_uri_contains_db_name(self):
        assert "airflow" in settings.airflow_db_uri
        assert "postgresql" in settings.airflow_db_uri

    def test_churnops_db_uri_contains_db_name(self):
        assert "churnops" in settings.churnops_db_uri
        assert "postgresql" in settings.churnops_db_uri

    # ── Postgres fields ───────────────────────────────────────────────────────
    def test_postgres_host_default(self):
        assert settings.postgres_host == "localhost"

    def test_postgres_port_default(self):
        assert settings.postgres_port == 5433
