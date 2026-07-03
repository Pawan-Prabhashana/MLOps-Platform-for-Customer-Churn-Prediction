# ruff: noqa: E402
"""build_dataset.py — CLI pipeline: raw CSV → cleaned → splits → parquet.

Usage
-----
    python pipelines/build_dataset.py
    python pipelines/build_dataset.py --raw data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
    python pipelines/build_dataset.py --seed 123 --out data/processed

Idempotent: existing parquet files are overwritten on each run.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Ensure src/ is on the path when the script is run directly
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from churnops.data.clean import clean_telco
from churnops.data.ingest import load_raw
from churnops.data.schema import TARGET_COL
from churnops.data.split import make_splits
from churnops.data.validate import validate
from churnops.logging import setup_logging

setup_logging()
logger = logging.getLogger("churnops.pipeline.build_dataset")


def _load_data_config() -> dict:
    cfg_path = _REPO_ROOT / "configs" / "data.yaml"
    with cfg_path.open() as f:
        return yaml.safe_load(f)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    cfg = _load_data_config()
    default_raw = str(_REPO_ROOT / cfg["paths"]["raw_csv"])
    default_out = str(_REPO_ROOT / cfg["paths"]["processed_dir"])
    default_seed = cfg["splits"]["seed"]

    p = argparse.ArgumentParser(
        description="Build cleaned train/val/test parquet splits from raw Telco CSV."
    )
    p.add_argument("--raw", default=default_raw, help="Path to raw CSV file")
    p.add_argument("--out", default=default_out, help="Output directory for parquet files")
    p.add_argument("--seed", type=int, default=default_seed, help="Random seed")
    p.add_argument(
        "--train-ratio", type=float, default=cfg["splits"]["train"], help="Train fraction"
    )
    p.add_argument(
        "--val-ratio", type=float, default=cfg["splits"]["val"], help="Val fraction"
    )
    p.add_argument(
        "--test-ratio", type=float, default=cfg["splits"]["test"], help="Test fraction"
    )
    return p.parse_args(argv)


def run(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    logger.info("Loading raw data from: %s", args.raw)
    raw_df = load_raw(args.raw)

    # ── Clean ─────────────────────────────────────────────────────────────────
    logger.info("Cleaning...")
    clean_df = clean_telco(raw_df)

    # ── Validate ──────────────────────────────────────────────────────────────
    logger.info("Validating...")
    validate(clean_df)

    # ── Split ─────────────────────────────────────────────────────────────────
    logger.info(
        "Splitting (train=%.0f%% / val=%.0f%% / test=%.0f%%, seed=%d)...",
        args.train_ratio * 100,
        args.val_ratio * 100,
        args.test_ratio * 100,
        args.seed,
    )
    splits = make_splits(
        clean_df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # ── Write parquet ─────────────────────────────────────────────────────────
    cfg = _load_data_config()["paths"]
    files = {
        "train": out_dir / cfg["train_file"],
        "val": out_dir / cfg["val_file"],
        "test": out_dir / cfg["test_file"],
    }
    split_dfs = {"train": splits.train, "val": splits.val, "test": splits.test}

    for name, path in files.items():
        split_dfs[name].to_parquet(path, index=False)
        logger.info("Wrote %s", path)

    # ── Summary ───────────────────────────────────────────────────────────────
    total = sum(len(d) for d in split_dfs.values())
    print("\n" + "=" * 58)
    print(f"{'Split':<8}  {'Rows':>6}  {'Churn rate':>11}  {'Output file'}")
    print("-" * 58)
    for name, df in split_dfs.items():
        rate = df[TARGET_COL].mean() * 100
        fname = files[name].name
        print(f"{name:<8}  {len(df):>6}  {rate:>10.1f}%  {fname}")
    print("-" * 58)
    total_rate = clean_df[TARGET_COL].mean() * 100
    print(f"{'total':<8}  {total:>6}  {total_rate:>10.1f}%  (full dataset)")
    print("=" * 58)
    print(f"\nParquet files written to: {out_dir.resolve()}")


if __name__ == "__main__":
    run()
