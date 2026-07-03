"""Central Settings object for churnops.

Loads non-secrets from configs/app.yaml and secrets from .env (or environment).
Import `settings` for direct access or call `get_settings()` for a cached accessor.
"""

from __future__ import annotations

import functools
from pathlib import Path

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve repo root regardless of working directory
_REPO_ROOT = Path(__file__).parent.parent.parent
_APP_YAML = _REPO_ROOT / "configs" / "app.yaml"


def _load_yaml_defaults() -> dict:
    """Load configs/app.yaml, return flat dict of env-var-style overrides."""
    if not _APP_YAML.exists():
        return {}
    with _APP_YAML.open() as f:
        data = yaml.safe_load(f) or {}
    return data


_yaml = _load_yaml_defaults()
_project = _yaml.get("project", {})
_kafka = _yaml.get("kafka", {})
_kafka_topics = _kafka.get("topics", {})
_mlflow = _yaml.get("mlflow", {})
_postgres = _yaml.get("postgres", {})


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Project paths ────────────────────────────────────────────────────────
    raw_data_dir: Path = Field(
        default=_REPO_ROOT / _project.get("raw_data_dir", "data/raw")
    )
    processed_data_dir: Path = Field(
        default=_REPO_ROOT / _project.get("processed_data_dir", "data/processed")
    )
    artifacts_dir: Path = Field(
        default=_REPO_ROOT / _project.get("artifacts_dir", "artifacts")
    )

    # ── Reproducibility ──────────────────────────────────────────────────────
    random_seed: int = Field(default=_yaml.get("random_seed", 42))

    # ── Kafka ────────────────────────────────────────────────────────────────
    kafka_bootstrap_servers: str = Field(
        default=_kafka.get("bootstrap_servers", "localhost:9092")
    )
    kafka_topic_raw_customers: str = Field(
        default=_kafka_topics.get("raw_customers", "telco.raw.customers")
    )
    kafka_topic_predictions: str = Field(
        default=_kafka_topics.get("predictions", "telco.churn.predictions")
    )
    kafka_topic_deadletter: str = Field(
        default=_kafka_topics.get("deadletter", "telco.deadletter")
    )

    # ── MLflow ───────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = Field(
        default=_mlflow.get("tracking_uri", "http://localhost:3000")
    )

    # ── Postgres ─────────────────────────────────────────────────────────────
    postgres_host: str = Field(default=_postgres.get("host", "localhost"))
    postgres_port: int = Field(default=_postgres.get("port", 5433))
    postgres_user: str = Field(default="churnops")
    postgres_password: str = Field(default="churnops_secret")
    postgres_mlflow_db: str = Field(default=_postgres.get("mlflow_db", "mlflow"))
    postgres_airflow_db: str = Field(default=_postgres.get("airflow_db", "airflow"))
    postgres_churnops_db: str = Field(default=_postgres.get("churnops_db", "churnops"))

    # ── Derived connection URIs ───────────────────────────────────────────────
    @property
    def mlflow_db_uri(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_mlflow_db}"
        )

    @property
    def airflow_db_uri(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_airflow_db}"
        )

    @property
    def churnops_db_uri(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_churnops_db}"
        )

    @field_validator("raw_data_dir", "processed_data_dir", "artifacts_dir", mode="before")
    @classmethod
    def _coerce_path(cls, v: object) -> Path:
        return Path(v)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance (re-reads on first call only)."""
    return Settings()


settings: Settings = get_settings()
