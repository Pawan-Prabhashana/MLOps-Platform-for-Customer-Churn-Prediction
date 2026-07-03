"""Thin factory for confluent-kafka Producer instances.

All settings read from configs/kafka.yaml — nothing is hardcoded here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from confluent_kafka import Producer

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_KAFKA_YAML = _REPO_ROOT / "configs" / "kafka.yaml"


def load_kafka_config() -> dict[str, Any]:
    """Load and return the parsed configs/kafka.yaml."""
    with _KAFKA_YAML.open() as f:
        return yaml.safe_load(f)


def build_producer(
    bootstrap_servers: str,
    *,
    acks: str = "all",
    linger_ms: int = 5,
    compression_type: str = "gzip",
    extra: dict[str, Any] | None = None,
) -> Producer:
    """Construct and return a configured confluent-kafka Producer.

    Args:
        bootstrap_servers: Comma-separated host:port list.
        acks:              Delivery acknowledgement level ("all", "1", "0").
        linger_ms:         Batching window in milliseconds.
        compression_type:  Message compression codec ("gzip", "snappy", "none").
        extra:             Any additional confluent-kafka config keys to merge.

    Returns:
        A ready-to-use :class:`confluent_kafka.Producer`.
    """
    conf: dict[str, Any] = {
        "bootstrap.servers": bootstrap_servers,
        "acks": acks,
        "linger.ms": linger_ms,
        "compression.type": compression_type,
        # Surface delivery errors immediately rather than silently dropping.
        "enable.idempotence": acks == "all",
    }
    if extra:
        conf.update(extra)

    logger.debug(
        "Building Kafka producer: servers=%s acks=%s linger=%sms compression=%s",
        bootstrap_servers,
        acks,
        linger_ms,
        compression_type,
    )
    return Producer(conf)


def producer_from_config(bootstrap_servers: str | None = None) -> Producer:
    """Build a Producer using settings from configs/kafka.yaml.

    The bootstrap_servers parameter (or value from app config) overrides the
    config-file default so the CLI flag flows through cleanly.
    """
    kafka_cfg = load_kafka_config()
    p_cfg = kafka_cfg.get("producer", {})

    from churnops.config import get_settings

    servers = bootstrap_servers or get_settings().kafka_bootstrap_servers
    return build_producer(
        servers,
        acks=p_cfg.get("acks", "all"),
        linger_ms=int(p_cfg.get("linger_ms", 5)),
        compression_type=p_cfg.get("compression_type", "gzip"),
    )
