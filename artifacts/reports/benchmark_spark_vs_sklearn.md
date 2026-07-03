# Spark MLlib vs scikit-learn — Performance Analysis

> **Reading these numbers.** This dataset is tiny (~7k rows), so Spark's JVM startup, serialization, and task-scheduling overhead typically make MLlib **slower** than scikit-learn here. That is expected and not a defect: the Spark path exists to scale *horizontally* to datasets that do not fit in memory on a single machine. On a toy dataset, sklearn wins on speed; at 10–100M+ rows the trade-off flips. Predictive metrics (ROC-AUC / PR-AUC / F1) should land in the same ballpark across both frameworks, confirming the pipelines are equivalent.

| Model | Framework | Estimator | Train (s) | Infer (s) | ROC-AUC | PR-AUC | F1 |
|-------|-----------|-----------|-----------|-----------|---------|--------|-----|
| logistic_regression | sklearn | LogisticRegression | 0.0198 | 0.0044 | 0.8447 | 0.6589 | 0.6222 |
| logistic_regression | spark | LogisticRegression | 2.8039 | 0.1829 | 0.8441 | 0.6534 | 0.6241 |
| random_forest | sklearn | RandomForestClassifier | 0.1701 | 0.0410 | 0.8194 | 0.6173 | 0.6014 |
| random_forest | spark | RandomForestClassifier | 6.3817 | 0.5421 | 0.8381 | 0.6631 | 0.6236 |
| gradient_boosting | sklearn | HistGradientBoostingClassifier | 0.4466 | 0.0159 | 0.8264 | 0.6523 | 0.6190 |
| gradient_boosting | spark | GBTClassifier | 9.6675 | 0.2056 | 0.8294 | 0.6537 | 0.6133 |
