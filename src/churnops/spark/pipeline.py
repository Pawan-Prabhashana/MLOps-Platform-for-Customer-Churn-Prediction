"""Assemble an MLlib Pipeline from the feature stages + a classifier estimator."""

from __future__ import annotations

import importlib
from typing import Any

from pyspark.ml import Pipeline
from pyspark.ml.classification import Classifier

from churnops.data.schema import TARGET_COL
from churnops.spark.preprocessing import FEATURES_COL, build_feature_stages


def build_estimator(model_key: str, cfg: dict, seed: int, weight_col: str | None = None) -> Classifier:
    """Instantiate an MLlib classifier from its dotted class path + params.

    Args:
        model_key:   Key from configs/spark.yaml `models`.
        cfg:         Parsed configs/spark.yaml.
        seed:        Random seed (from settings.random_seed).
        weight_col:  Optional column name for instance weights (class balancing).

    Returns:
        An unfitted MLlib Classifier with featuresCol/labelCol wired up.
    """
    model_cfg = cfg["models"][model_key]
    module_path, cls_name = model_cfg["class"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), cls_name)

    params: dict[str, Any] = dict(model_cfg.get("params", {}))
    params["featuresCol"] = FEATURES_COL
    params["labelCol"] = TARGET_COL
    params["predictionCol"] = "prediction"

    estimator = cls(**params)

    # Not all MLlib estimators accept these params (e.g. LogisticRegression has
    # no `seed`; GBTClassifier has no `weightCol`). Set them only when supported.
    if estimator.hasParam("seed"):
        estimator = estimator.setSeed(seed)
    if weight_col is not None and estimator.hasParam("weightCol"):
        estimator = estimator.setWeightCol(weight_col)

    return estimator


def build_pipeline(estimator: Classifier) -> Pipeline:
    """Return an unfitted MLlib Pipeline = feature stages + estimator."""
    stages = build_feature_stages()
    stages.append(estimator)
    return Pipeline(stages=stages)
