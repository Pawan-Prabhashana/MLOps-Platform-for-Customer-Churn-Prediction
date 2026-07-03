"""Tests for the PySpark MLlib path.

Skipped automatically if PySpark or a JVM (java) is not available, but written
to actually exercise the pipeline when they are. Kept fast: tiny samples and
small models so the suite stays quick.
"""

from __future__ import annotations

import functools
import shutil
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).parent.parent


def _spark_available() -> bool:
    try:
        import pyspark  # noqa: F401
    except ImportError:
        return False
    return shutil.which("java") is not None


pytestmark = pytest.mark.skipif(
    not _spark_available(),
    reason="PySpark or a JVM (java) is not available in this environment.",
)


@functools.lru_cache(maxsize=1)
def _rdd_pickling_works() -> bool:
    """Probe whether Python RDD closure pickling works on this interpreter.

    PySpark 3.5.x bundles a cloudpickle that infinitely recurses when pickling
    Python closures on Python >= 3.12 (distutils/typing internals changed). This
    breaks anything RDD-based, including PipelineModel.load() (which reads model
    metadata via an internal sc.textFile RDD). DataFrame/MLlib train + evaluate +
    model.save() are unaffected because they run entirely JVM-side.
    """
    from churnops.spark.session import get_spark

    spark = get_spark("rdd-probe")
    try:
        spark.sparkContext.parallelize([1, 2, 3]).map(lambda x: x + 1).collect()
        return True
    except BaseException:  # noqa: BLE001
        return False


def _threshold() -> float:
    cfg = yaml.safe_load((_REPO_ROOT / "configs" / "spark.yaml").read_text())
    return cfg["thresholds"]["roc_auc"]


@pytest.fixture(scope="module")
def spark():
    from churnops.spark.session import get_spark, stop_spark

    session = get_spark("churnops-test")
    yield session
    stop_spark()


# ── Session ───────────────────────────────────────────────────────────────────

def test_spark_session_builds(spark):
    assert spark is not None
    assert spark.version.startswith("3.5")


def test_read_split(spark):
    from churnops.spark.session import read_split

    df = read_split(spark, "train")
    assert df.count() > 0
    from churnops.data.schema import TARGET_COL

    assert TARGET_COL in df.columns


# ── Preprocessing + pipeline fit on a small sample ────────────────────────────

def test_pipeline_fits_on_sample(spark):
    from pyspark.ml.classification import LogisticRegression

    from churnops.data.schema import TARGET_COL
    from churnops.spark.pipeline import build_pipeline
    from churnops.spark.preprocessing import FEATURES_COL
    from churnops.spark.session import read_split

    sample = read_split(spark, "train").sample(fraction=0.1, seed=42)
    estimator = LogisticRegression(
        featuresCol=FEATURES_COL, labelCol=TARGET_COL, maxIter=10
    )
    pipeline = build_pipeline(estimator)
    model = pipeline.fit(sample)
    preds = model.transform(read_split(spark, "val").limit(50))
    cols = preds.columns
    assert "prediction" in cols
    assert "probability" in cols


# ── Metric threshold ──────────────────────────────────────────────────────────

def test_logreg_clears_roc_threshold(spark):
    from churnops.spark.train import train

    _, metrics = train(model_key="logistic_regression", spark=spark)
    assert metrics["test"]["roc_auc"] >= _threshold(), (
        f"test ROC-AUC {metrics['test']['roc_auc']} below threshold {_threshold()}"
    )


# ── Save → reload → identical predictions ─────────────────────────────────────

def test_saved_reloaded_model_predicts_identically(spark, tmp_path):
    if not _rdd_pickling_works():
        pytest.skip(
            "PipelineModel.load() needs Python RDD closure pickling, which is "
            "broken on PySpark 3.5.x + Python >= 3.12 (bundled cloudpickle "
            "infinite recursion). Training, evaluation and model.save() still "
            "work; only Python-side model loading is affected on this interpreter."
        )

    from pyspark.ml import PipelineModel
    from pyspark.ml.classification import LogisticRegression

    from churnops.data.schema import TARGET_COL
    from churnops.spark.pipeline import build_pipeline
    from churnops.spark.preprocessing import FEATURES_COL
    from churnops.spark.session import read_split

    sample = read_split(spark, "train").sample(fraction=0.2, seed=42)
    estimator = LogisticRegression(featuresCol=FEATURES_COL, labelCol=TARGET_COL, maxIter=20)
    model = build_pipeline(estimator).fit(sample)

    test_sample = read_split(spark, "test").limit(30).cache()
    before = [r["prediction"] for r in model.transform(test_sample).select("prediction").collect()]

    save_path = tmp_path / "spark_model"
    model.write().overwrite().save(str(save_path))
    reloaded = PipelineModel.load(str(save_path))
    after = [r["prediction"] for r in reloaded.transform(test_sample).select("prediction").collect()]

    assert before == after


# ── Unseen categorical value must not crash (handleInvalid="keep") ────────────

def test_unseen_category_does_not_crash(spark):
    from pyspark.ml.classification import LogisticRegression

    from churnops.data.schema import TARGET_COL
    from churnops.spark.pipeline import build_pipeline
    from churnops.spark.preprocessing import FEATURES_COL
    from churnops.spark.session import read_split

    train_sample = read_split(spark, "train").sample(fraction=0.2, seed=42)
    estimator = LogisticRegression(featuresCol=FEATURES_COL, labelCol=TARGET_COL, maxIter=10)
    model = build_pipeline(estimator).fit(train_sample)

    # Take one real row and inject an unseen categorical value.
    one = read_split(spark, "test").limit(1)
    mutated = one.withColumn("Contract", one["Contract"].cast("string"))
    from pyspark.sql import functions as F

    mutated = mutated.withColumn("Contract", F.lit("Quantum-entangled 999yr plan"))

    preds = model.transform(mutated)
    # Should produce exactly one prediction without throwing.
    assert preds.select("prediction").count() == 1
