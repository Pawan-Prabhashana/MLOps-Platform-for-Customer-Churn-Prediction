"""Build a full sklearn Pipeline: preprocessor → estimator.

Usage
-----
    estimator = LogisticRegression(class_weight="balanced", random_state=42)
    pipe = build_pipeline(estimator)
    pipe.fit(X_train, y_train)
    pipe.predict_proba(X_new)
"""

from __future__ import annotations

from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline

from churnops.models.preprocessing import build_preprocessor


def build_pipeline(estimator: ClassifierMixin) -> Pipeline:
    """Return an unfitted Pipeline(preprocess → model).

    Args:
        estimator: Any sklearn-compatible classifier.

    Returns:
        sklearn Pipeline with steps ["preprocess", "model"].
    """
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )
