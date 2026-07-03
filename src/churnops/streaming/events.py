"""Dataset loading and batch-mode checkpoint helpers.

Checkpoint format (JSON):
    {
        "last_index": 1500,      # absolute row index — the *next* row to send
        "total_sent": 1500,      # cumulative rows sent across all sessions
        "finished": false        # true once the full dataset (or limit) is done
    }

The checkpoint tracks an absolute position into the full DataFrame so that
resuming after a crash simply picks up where the last successful flush left off
— no rows are re-sent and no rows are skipped.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the raw Telco CSV with minimal preprocessing.

    The only transformation applied: ``TotalCharges`` blank strings are parsed
    as ``NaN`` (handled automatically by pandas' numeric coercion with
    ``errors='coerce'``).  Everything else is left as-is so the consumer/model
    pipeline sees raw-ish events.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    # Coerce TotalCharges to float; blanks become NaN → serialized as JSON null.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    logger.debug("Dataset loaded: %d rows from %s", len(df), path)
    return df


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def checkpoint_load(path: str | Path) -> dict:
    """Load checkpoint state from disk, or return a fresh zero state."""
    p = Path(path)
    if p.exists():
        state = json.loads(p.read_text())
        logger.info(
            "Checkpoint loaded: last_index=%d  total_sent=%d  finished=%s",
            state.get("last_index", 0),
            state.get("total_sent", 0),
            state.get("finished", False),
        )
        return state
    return {"last_index": 0, "total_sent": 0, "finished": False}


def checkpoint_save(path: str | Path, last_index: int, total_sent: int, finished: bool = False) -> None:
    """Persist checkpoint state to disk (atomic write via temp-then-rename style)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    state = {"last_index": last_index, "total_sent": total_sent, "finished": finished}
    p.write_text(json.dumps(state, indent=2))
    logger.debug("Checkpoint saved: last_index=%d  total_sent=%d", last_index, total_sent)


def checkpoint_clear(path: str | Path) -> None:
    """Delete the checkpoint file (called on successful completion)."""
    p = Path(path)
    if p.exists():
        p.unlink()
        logger.debug("Checkpoint cleared: %s", p)


def checkpoint_reset(path: str | Path) -> None:
    """Reset checkpoint to row 0 (equivalent to starting fresh)."""
    checkpoint_save(path, last_index=0, total_sent=0, finished=False)
    logger.info("Checkpoint reset to row 0 at %s", path)
