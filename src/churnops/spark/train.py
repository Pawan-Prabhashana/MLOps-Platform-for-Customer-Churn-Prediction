"""Train + evaluate MLlib models on Spark.

Computes the SAME headline metrics as the sklearn path (accuracy, positive-class
precision/recall/f1, ROC-AUC, PR-AUC) so Spark vs sklearn numbers line up.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from churnops.config import get_settings
from churnops.data.schema import TARGET_COL
from churnops.spark.pipeline import build_estimator, build_pipeline
from churnops.spark.session import get_spark, load_spark_config, read_split

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent.parent.parent

WEIGHT_COL = "class_weight"


def _load_config() -> dict:
    return load_spark_config()


def add_balanced_weights(df: DataFrame) -> DataFrame:
    """Add a per-row balanced class-weight column (sklearn class_weight='balanced').

    weight(class c) = n_samples / (n_classes * n_samples_in_c)
    """
    counts = {row[TARGET_COL]: row["cnt"] for row in df.groupBy(TARGET_COL).agg(F.count("*").alias("cnt")).collect()}
    total = sum(counts.values())
    n_classes = len(counts)
    weights = {cls: total / (n_classes * cnt) for cls, cnt in counts.items()}

    weight_expr = F.when(F.col(TARGET_COL) == list(weights.keys())[0], F.lit(list(weights.values())[0]))
    for cls, w in list(weights.items())[1:]:
        weight_expr = weight_expr.when(F.col(TARGET_COL) == cls, F.lit(w))
    weight_expr = weight_expr.otherwise(F.lit(1.0))
    return df.withColumn(WEIGHT_COL, weight_expr)


def evaluate(predictions: DataFrame, split_name: str) -> dict[str, Any]:
    """Compute headline metrics from a predictions DataFrame."""
    bin_roc = BinaryClassificationEvaluator(
        labelCol=TARGET_COL, rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )
    bin_pr = BinaryClassificationEvaluator(
        labelCol=TARGET_COL, rawPredictionCol="rawPrediction", metricName="areaUnderPR"
    )

    def multi(metric: str, label: float | None = None) -> float:
        ev = MulticlassClassificationEvaluator(
            labelCol=TARGET_COL, predictionCol="prediction", metricName=metric
        )
        if label is not None:
            ev = ev.setMetricLabel(label)
        return float(ev.evaluate(predictions))

    n = predictions.count()
    metrics = {
        "split": split_name,
        "n_samples": n,
        "accuracy": round(multi("accuracy"), 4),
        "precision": round(multi("precisionByLabel", 1.0), 4),
        "recall": round(multi("recallByLabel", 1.0), 4),
        "f1": round(multi("fMeasureByLabel", 1.0), 4),
        "roc_auc": round(float(bin_roc.evaluate(predictions)), 4),
        "pr_auc": round(float(bin_pr.evaluate(predictions)), 4),
    }
    # 2x2 confusion matrix [[TN, FP], [FN, TP]]
    cm = (
        predictions.groupBy(TARGET_COL, "prediction")
        .count()
        .collect()
    )
    matrix = [[0, 0], [0, 0]]
    for row in cm:
        actual = int(row[TARGET_COL])
        pred = int(row["prediction"])
        matrix[actual][pred] = int(row["count"])
    metrics["confusion_matrix"] = matrix

    logger.info(
        "[%s] roc_auc=%.4f  pr_auc=%.4f  recall=%.4f  f1=%.4f",
        split_name.upper(),
        metrics["roc_auc"],
        metrics["pr_auc"],
        metrics["recall"],
        metrics["f1"],
    )
    return metrics


def train(
    model_key: str | None = None,
    spark: SparkSession | None = None,
    sample_fraction: float | None = None,
) -> tuple[PipelineModel, dict]:
    """Train an MLlib model on the train split, evaluate on val + test.

    Args:
        model_key:        Key from configs/spark.yaml `models` (default: config default).
        spark:            Optional existing SparkSession (default: cached get_spark()).
        sample_fraction:  If set, sample the train split (used by fast tests).

    Returns:
        (fitted PipelineModel, metrics dict) — same shape as the sklearn path.
    """
    cfg = _load_config()
    settings = get_settings()
    spark = spark or get_spark()

    if model_key is None:
        model_key = cfg["default_model"]
    if model_key not in cfg["models"]:
        raise ValueError(
            f"Unknown model key '{model_key}'. Available: {list(cfg['models'].keys())}"
        )

    train_df = read_split(spark, "train")
    val_df = read_split(spark, "val")
    test_df = read_split(spark, "test")

    if sample_fraction:
        train_df = train_df.sample(withReplacement=False, fraction=sample_fraction, seed=settings.random_seed)

    train_df = add_balanced_weights(train_df).cache()
    logger.info("Fitting spark:%s on %d training rows...", model_key, train_df.count())

    estimator = build_estimator(model_key, cfg, settings.random_seed, weight_col=WEIGHT_COL)
    pipeline = build_pipeline(estimator)
    model = pipeline.fit(train_df)
    logger.info("Training complete.")

    val_metrics = evaluate(model.transform(val_df), "val")
    test_metrics = evaluate(model.transform(test_df), "test")

    train_df.unpersist()

    metrics = {
        "model_key": model_key,
        "model_class": cfg["models"][model_key]["class"],
        "val": val_metrics,
        "test": test_metrics,
    }
    return model, metrics


def metrics_table(metrics: dict) -> str:
    """Return a formatted val + test metrics table (mirrors the sklearn one)."""
    header = f"\n{'Metric':<14} {'Val':>10} {'Test':>10}"
    sep = "-" * 38
    rows = [header, sep]
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        v = metrics["val"].get(k, "-")
        t = metrics["test"].get(k, "-")
        rows.append(f"{k:<14} {v:>10} {t:>10}")
    rows.append(sep)
    rows.append(f"\nVal  confusion matrix:\n{metrics['val']['confusion_matrix']}")
    rows.append(f"\nTest confusion matrix:\n{metrics['test']['confusion_matrix']}")
    return "\n".join(rows)
