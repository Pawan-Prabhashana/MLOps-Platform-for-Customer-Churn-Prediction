"""Dead-letter message construction and publishing.

A record is dead-lettered (routed to ``telco.deadletter``) whenever it cannot be
processed: JSON parse failure, missing required fields, type-coercion failure,
or a model-scoring error. The dead-letter VALUE preserves the original raw
message plus an ``error`` description and a ``failed_ts`` timestamp so the
failure can be triaged later without losing the payload.

    KEY   = customerID (or "unknown" if it couldn't be extracted)
    VALUE = {
        "original": <parsed dict OR raw string if unparseable>,
        "error":    "<why it failed>",
        "failed_ts":"2025-10-03T04:00:01Z"
    }
"""

from __future__ import annotations

import json
import logging
from typing import Any

from churnops.streaming.serialization import now_utc

logger = logging.getLogger(__name__)

UNKNOWN_KEY = "unknown"


def build_dead_letter(
    raw_value: bytes | str | dict,
    error: str,
    key: str | None = None,
) -> tuple[bytes, bytes]:
    """Build the (key_bytes, value_bytes) for a dead-letter record.

    The original payload is preserved as parsed JSON when possible, otherwise as
    the raw string, so nothing is ever lost.
    """
    key_str = key or UNKNOWN_KEY

    original: Any
    if isinstance(raw_value, dict):
        original = raw_value
    else:
        text = raw_value.decode("utf-8", errors="replace") if isinstance(raw_value, bytes) else raw_value
        try:
            original = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            original = text  # keep the unparseable payload verbatim

    payload = {
        "original": original,
        "error": error,
        "failed_ts": now_utc(),
    }
    return key_str.encode("utf-8"), json.dumps(payload, ensure_ascii=False).encode("utf-8")


def send_dead_letter(
    producer: Any,
    topic: str,
    raw_value: bytes | str | dict,
    error: str,
    key: str | None = None,
    on_delivery: Any = None,
) -> None:
    """Serialise and publish a dead-letter record; log a warning."""
    k, v = build_dead_letter(raw_value, error, key)
    producer.produce(topic, key=k, value=v, on_delivery=on_delivery)
    logger.warning("Dead-lettered record key=%s error=%s", k.decode(), error)
