"""churnops.spark — PySpark MLlib training path (parallel to churnops.models).

This package mirrors the sklearn path using Spark DataFrame + MLlib APIs so the
two implementations are directly comparable on the same processed parquet splits
and the same schema column lists. It never imports from or mutates the sklearn
path — the two are fully independent.
"""
