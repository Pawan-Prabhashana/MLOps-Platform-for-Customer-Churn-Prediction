"""Head-to-head benchmark: Spark MLlib path vs the scikit-learn path.

For each comparable model pair (sklearn estimator vs its MLlib counterpart) we
record, on the SAME processed test split:
  - training wall-clock time
  - inference wall-clock time on the test split
  - key metrics: ROC-AUC, PR-AUC, F1

Outputs a markdown table, a CSV, and two PNG charts (metrics + timing) into the
reports dir, and returns the rows so the CLI can log them to MLflow.

Honest note (also printed by the CLI and written into the report): on a dataset
this small (~7k rows) Spark's JVM + task-scheduling overhead usually makes it
SLOWER than sklearn. The value of the Spark path is horizontal scalability to
data that does not fit on one machine — not winning on a toy dataset.
"""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from churnops.config import get_settings  # noqa: E402
from churnops.spark import train as spark_train  # noqa: E402
from churnops.spark.session import get_spark, load_spark_config, read_split  # noqa: E402

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent.parent.parent


@dataclass
class BenchRow:
    model_pair: str          # e.g. "logistic_regression"
    framework: str           # "sklearn" | "spark"
    estimator: str           # concrete class name
    train_time_s: float
    infer_time_s: float
    roc_auc: float
    pr_auc: float
    f1: float


# ── sklearn side ──────────────────────────────────────────────────────────────

def _bench_sklearn(model_key: str) -> BenchRow:
    """Time + score a single sklearn model on the shared splits."""
    # Imported lazily so the Spark path never hard-depends on being run together.
    from churnops.models import train as sk_train

    cfg = sk_train._load_model_config()
    data_cfg = sk_train._load_data_config()
    proc_dir = _REPO_ROOT / data_cfg["paths"]["processed_dir"]

    train_df = pd.read_parquet(proc_dir / data_cfg["paths"]["train_file"])
    test_df = pd.read_parquet(proc_dir / data_cfg["paths"]["test_file"])
    X_train, y_train = sk_train._xy(train_df)
    X_test, y_test = sk_train._xy(test_df)

    estimator = sk_train._build_estimator(model_key, cfg)
    pipeline = sk_train.build_pipeline(estimator)

    fit_params: dict = {}
    if "HistGradient" in type(estimator).__name__:
        from sklearn.utils.class_weight import compute_sample_weight

        fit_params["model__sample_weight"] = compute_sample_weight("balanced", y_train)

    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train, **fit_params)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)
    infer_time = time.perf_counter() - t0

    return BenchRow(
        model_pair=model_key,
        framework="sklearn",
        estimator=type(estimator).__name__,
        train_time_s=round(train_time, 4),
        infer_time_s=round(infer_time, 4),
        roc_auc=round(float(roc_auc_score(y_test, y_proba)), 4),
        pr_auc=round(float(average_precision_score(y_test, y_proba)), 4),
        f1=round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
    )


# ── Spark side ────────────────────────────────────────────────────────────────

def _bench_spark(model_key: str) -> BenchRow:
    """Time + score a single MLlib model on the shared splits."""
    from churnops.spark.pipeline import build_estimator, build_pipeline

    cfg = load_spark_config()
    settings = get_settings()
    spark = get_spark()

    train_df = spark_train.add_balanced_weights(read_split(spark, "train")).cache()
    train_df.count()  # force cache/materialize so fit time excludes the read
    test_df = read_split(spark, "test").cache()
    test_df.count()

    estimator = build_estimator(model_key, cfg, settings.random_seed, weight_col=spark_train.WEIGHT_COL)
    pipeline = build_pipeline(estimator)

    t0 = time.perf_counter()
    model = pipeline.fit(train_df)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    predictions = model.transform(test_df)
    # Force materialization so we measure real inference, not lazy planning.
    predictions.select("prediction", "rawPrediction").collect()
    infer_time = time.perf_counter() - t0

    metrics = spark_train.evaluate(predictions, "test")
    train_df.unpersist()
    test_df.unpersist()

    return BenchRow(
        model_pair=cfg["models"][model_key]["sklearn_counterpart"],
        framework="spark",
        estimator=cfg["models"][model_key]["class"].rsplit(".", 1)[-1],
        train_time_s=round(train_time, 4),
        infer_time_s=round(infer_time, 4),
        roc_auc=metrics["roc_auc"],
        pr_auc=metrics["pr_auc"],
        f1=metrics["f1"],
    )


# ── Orchestration ─────────────────────────────────────────────────────────────

def run_benchmark(spark_model_keys: list[str] | None = None) -> list[BenchRow]:
    """Benchmark each Spark model against its sklearn counterpart.

    Returns a flat list of BenchRow (sklearn row + spark row per pair).
    """
    cfg = load_spark_config()
    keys = spark_model_keys or list(cfg["models"].keys())

    rows: list[BenchRow] = []
    for spark_key in keys:
        sklearn_key = cfg["models"][spark_key]["sklearn_counterpart"]
        logger.info("Benchmarking pair: sklearn=%s  vs  spark=%s", sklearn_key, spark_key)
        rows.append(_bench_sklearn(sklearn_key))
        rows.append(_bench_spark(spark_key))
    return rows


# ── Reporting ─────────────────────────────────────────────────────────────────

COMMENTARY = (
    "> **Reading these numbers.** This dataset is tiny (~7k rows), so Spark's JVM "
    "startup, serialization, and task-scheduling overhead typically make MLlib "
    "**slower** than scikit-learn here. That is expected and not a defect: the Spark "
    "path exists to scale *horizontally* to datasets that do not fit in memory on a "
    "single machine. On a toy dataset, sklearn wins on speed; at 10–100M+ rows the "
    "trade-off flips. Predictive metrics (ROC-AUC / PR-AUC / F1) should land in the "
    "same ballpark across both frameworks, confirming the pipelines are equivalent."
)


def _reports_dir() -> Path:
    cfg = load_spark_config()
    d = _REPO_ROOT / cfg["artifacts"]["reports_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_csv(rows: list[BenchRow], path: Path | None = None) -> Path:
    path = path or (_reports_dir() / "benchmark_spark_vs_sklearn.csv")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))
    logger.info("Benchmark CSV → %s", path)
    return path


def write_markdown(rows: list[BenchRow], path: Path | None = None) -> Path:
    path = path or (_reports_dir() / "benchmark_spark_vs_sklearn.md")
    lines = [
        "# Spark MLlib vs scikit-learn — Performance Analysis",
        "",
        COMMENTARY,
        "",
        "| Model | Framework | Estimator | Train (s) | Infer (s) | ROC-AUC | PR-AUC | F1 |",
        "|-------|-----------|-----------|-----------|-----------|---------|--------|-----|",
    ]
    for r in rows:
        lines.append(
            f"| {r.model_pair} | {r.framework} | {r.estimator} | "
            f"{r.train_time_s:.4f} | {r.infer_time_s:.4f} | "
            f"{r.roc_auc:.4f} | {r.pr_auc:.4f} | {r.f1:.4f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))
    logger.info("Benchmark markdown → %s", path)
    return path


def _pairs(rows: list[BenchRow]) -> list[str]:
    seen: list[str] = []
    for r in rows:
        if r.model_pair not in seen:
            seen.append(r.model_pair)
    return seen


def plot_metrics(rows: list[BenchRow], path: Path | None = None) -> Path:
    """Grouped bar chart of ROC-AUC / PR-AUC / F1, sklearn vs spark per pair."""
    path = path or (_reports_dir() / "benchmark_metrics.png")
    pairs = _pairs(rows)
    metrics = ["roc_auc", "pr_auc", "f1"]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, 10), sharex=True)
    x = range(len(pairs))
    width = 0.38

    for ax, metric in zip(axes, metrics, strict=False):
        sk_vals = [_lookup(rows, p, "sklearn", metric) for p in pairs]
        sp_vals = [_lookup(rows, p, "spark", metric) for p in pairs]
        ax.bar([i - width / 2 for i in x], sk_vals, width, label="sklearn", color="#4C72B0")
        ax.bar([i + width / 2 for i in x], sp_vals, width, label="spark", color="#DD8452")
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        for i, (s, p_) in enumerate(zip(sk_vals, sp_vals, strict=False)):
            ax.text(i - width / 2, s + 0.01, f"{s:.3f}", ha="center", fontsize=7)
            ax.text(i + width / 2, p_ + 0.01, f"{p_:.3f}", ha="center", fontsize=7)
        ax.legend(loc="lower right", fontsize=8)

    axes[-1].set_xticks(list(x))
    axes[-1].set_xticklabels(pairs, rotation=15)
    axes[0].set_title("Predictive metrics: scikit-learn vs Spark MLlib")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Metrics chart → %s", path)
    return path


def plot_timing(rows: list[BenchRow], path: Path | None = None) -> Path:
    """Grouped bar chart of train + inference time, sklearn vs spark per pair."""
    path = path or (_reports_dir() / "benchmark_timing.png")
    pairs = _pairs(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = range(len(pairs))
    width = 0.38

    for ax, field, title in (
        (axes[0], "train_time_s", "Training time (s)"),
        (axes[1], "infer_time_s", "Inference time (s)"),
    ):
        sk_vals = [_lookup(rows, p, "sklearn", field) for p in pairs]
        sp_vals = [_lookup(rows, p, "spark", field) for p in pairs]
        ax.bar([i - width / 2 for i in x], sk_vals, width, label="sklearn", color="#4C72B0")
        ax.bar([i + width / 2 for i in x], sp_vals, width, label="spark", color="#DD8452")
        ax.set_title(title)
        ax.set_xticks(list(x))
        ax.set_xticklabels(pairs, rotation=15)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend()
    fig.suptitle("Wall-clock timing: scikit-learn vs Spark MLlib (lower is better)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Timing chart → %s", path)
    return path


def _lookup(rows: list[BenchRow], pair: str, framework: str, field: str) -> float:
    for r in rows:
        if r.model_pair == pair and r.framework == framework:
            return float(getattr(r, field))
    return 0.0


def render_table(rows: list[BenchRow]) -> str:
    """Return a plaintext table for console output."""
    header = (
        f"{'Model':<20} {'Framework':<9} {'Train(s)':>10} {'Infer(s)':>10} "
        f"{'ROC-AUC':>9} {'PR-AUC':>8} {'F1':>7}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"{r.model_pair:<20} {r.framework:<9} {r.train_time_s:>10.4f} "
            f"{r.infer_time_s:>10.4f} {r.roc_auc:>9.4f} {r.pr_auc:>8.4f} {r.f1:>7.4f}"
        )
    return "\n".join(lines)
