# ruff: noqa: E402
"""predict_sklearn.py — CLI: score new records through the saved Pipeline.

Usage
-----
    # Score a CSV file
    python pipelines/predict_sklearn.py --input sample.csv

    # Score a JSON-lines file  (one JSON object per line)
    python pipelines/predict_sklearn.py --input sample.jsonl --format jsonl

    # Write predictions to a file instead of stdout
    python pipelines/predict_sklearn.py --input sample.csv --output preds.csv

    # Use a non-default pipeline artifact
    python pipelines/predict_sklearn.py --input sample.csv --pipeline artifacts/sklearn/pipeline.joblib
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import pandas as pd

from churnops.logging import setup_logging
from churnops.models.inference import predict
from churnops.models.persistence import load_pipeline

setup_logging()


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Score records using the saved churnops Pipeline."
    )
    p.add_argument("--input", required=True, help="Input CSV or JSON-lines file")
    p.add_argument(
        "--format",
        choices=["csv", "jsonl"],
        default=None,
        help="Input format. Inferred from extension if omitted.",
    )
    p.add_argument(
        "--pipeline",
        default=None,
        help="Path to pipeline.joblib (default: from configs/model.yaml)",
    )
    p.add_argument("--output", default=None, help="Write predictions CSV here (default: stdout)")
    return p.parse_args(argv)


def _load_input(path: Path, fmt: str | None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    resolved_fmt = fmt or ("jsonl" if suffix in {".jsonl", ".ndjson"} else "csv")
    if resolved_fmt == "jsonl":
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return pd.DataFrame(records)
    return pd.read_csv(path)


def run(argv=None) -> None:
    args = parse_args(argv)
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    pipeline = load_pipeline(args.pipeline)
    df = _load_input(input_path, args.format)
    print(f"Loaded {len(df)} rows from {input_path}", file=sys.stderr)

    results = predict(df, pipeline=pipeline)

    out_rows = [
        {
            "customerID": r.customer_id,
            "prediction": r.prediction,
            "label": r.label,
            "churn_probability": r.churn_probability,
        }
        for r in results
    ]
    out_df = pd.DataFrame(out_rows)

    if args.output:
        out_df.to_csv(args.output, index=False)
        print(f"Predictions written to {args.output}", file=sys.stderr)
    else:
        print(out_df.to_csv(index=False))


if __name__ == "__main__":
    run()
