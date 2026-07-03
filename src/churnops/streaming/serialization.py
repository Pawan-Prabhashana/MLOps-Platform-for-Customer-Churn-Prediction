"""Row → Kafka message serialization helpers.

Responsibilities:
  - Extract the Kafka message KEY (customerID, UTF-8 bytes).
  - Inject the ``event_ts`` field (ISO-8601 UTC).
  - Convert a raw DataFrame row dict to JSON bytes, preserving dtype semantics:
      * int columns  → JSON number (integer)
      * float columns → JSON number (float), NaN → JSON null
      * string columns → JSON string
  - Handle numpy scalar types so json.dumps never raises.

The producer emits *raw-ish* events — only the minimal coercion needed to
produce valid JSON is applied here. Heavy feature cleaning (e.g. Yes/No →
0/1) is left to the model pipeline on the consumer side.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from typing import Any

# ── Timestamp ─────────────────────────────────────────────────────────────────

def now_utc() -> str:
    """Return the current UTC time as an ISO-8601 string ending in 'Z'."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Type coercion ─────────────────────────────────────────────────────────────

def _coerce(value: Any) -> Any:
    """Convert a single value to a JSON-safe Python native type.

    - numpy integer scalars  → int
    - numpy float scalars    → float  (NaN → None so the JSON is 'null')
    - native float NaN/Inf   → None
    - everything else        → unchanged
    """
    # Numpy scalar detection without hard-importing numpy at module level.
    type_name = type(value).__module__
    if type_name == "numpy":
        cls = type(value).__name__
        if "int" in cls:
            return int(value)
        if "float" in cls:
            f = float(value)
            return None if math.isnan(f) or math.isinf(f) else f
    # Native Python float NaN/Inf.
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


# ── Public API ────────────────────────────────────────────────────────────────

def extract_key(row: dict[str, Any]) -> bytes:
    """Return the Kafka message key: customerID encoded as UTF-8 bytes.

    All events for the same customer are guaranteed to land on the same
    partition (topic-level ordering per customer).
    """
    return str(row["customerID"]).encode("utf-8")


def row_to_bytes(row: dict[str, Any], event_ts: str | None = None) -> bytes:
    """Serialise a raw dataset row to UTF-8-encoded JSON bytes.

    Args:
        row:       A dict from ``df.iloc[i].to_dict()`` (may contain numpy scalars
                   and NaN values for blank TotalCharges entries).
        event_ts:  ISO-8601 UTC string to use as the ``event_ts`` field.
                   If ``None``, the current UTC wall-clock time is used.

    Returns:
        UTF-8 JSON bytes matching the message contract exactly.
    """
    msg: dict[str, Any] = {k: _coerce(v) for k, v in row.items()}
    msg["event_ts"] = event_ts or now_utc()
    return json.dumps(msg, ensure_ascii=False).encode("utf-8")


def bytes_to_dict(data: bytes) -> dict[str, Any]:
    """Deserialise a Kafka message value back to a Python dict (for testing)."""
    return json.loads(data.decode("utf-8"))


# ── Expected message shape (used in tests and docs) ──────────────────────────

EXPECTED_FIELDS = frozenset({
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
    "event_ts",
})
