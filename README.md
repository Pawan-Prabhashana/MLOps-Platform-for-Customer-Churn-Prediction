# churnops

End-to-end **Telco Customer Churn MLOps** platform — monorepo containing the
feature engineering pipelines, model training, serving, and monitoring code,
backed by a fully Dockerized local development stack.

## Quick start

```bash
# 1. Clone and enter the repo
git clone https://github.com/Pawan-Prabhashana/MLOps-Platform-for-Customer-Churn-Prediction.git
cd MLOps-Platform-for-Customer-Churn-Prediction

# 2. Copy and edit secrets
cp .env.example .env   # fill in real values (or leave defaults for local dev)

# 3. Install the Python package (needs Python 3.11+)
make install           # pip install -e ".[dev]"

# 4. Start the local stack (Postgres, Kafka, MLflow, Airflow)
make up                # docker compose up -d

# 5. Verify services
#   MLflow UI  → http://localhost:3000
#   Airflow UI → http://localhost:8080  (admin / admin)
```

## Services

| Service    | Host port | Notes                                    |
|------------|-----------|------------------------------------------|
| PostgreSQL | 5432      | Databases: `mlflow`, `airflow`, `churnops` |
| Kafka      | 9092      | KRaft mode, 3 topics pre-created         |
| MLflow     | **3000**  | Backed by Postgres `mlflow` DB           |
| Airflow    | 8080      | LocalExecutor, DAGs from `./dags/`       |

## Kafka topics

| Topic                     | Purpose                         |
|---------------------------|---------------------------------|
| `telco.raw.customers`     | Inbound raw customer events     |
| `telco.churn.predictions` | Model prediction output         |
| `telco.deadletter`        | Un-processable / failed events  |

## Project layout

```
churnops/
├── src/churnops/       # installable Python package
│   ├── config.py       # pydantic-settings Settings object
│   └── logging.py      # logging setup helper
├── pipelines/          # CLI pipeline entrypoints
├── dags/               # Airflow DAGs
├── notebooks/          # exploration notebooks
├── tests/              # pytest suite
├── configs/            # app.yaml (non-secrets), logging.yaml
├── docker/             # Dockerfile.mlflow, init.sql
├── data/raw/           # raw input data (untracked)
├── data/processed/     # derived features  (untracked)
├── docker-compose.yml
├── Makefile
└── pyproject.toml
```

## Useful Makefile targets

| Target     | Action                               |
|------------|--------------------------------------|
| `make up`  | Start all Docker services            |
| `make down`| Stop and remove containers           |
| `make logs`| Tail service logs                    |
| `make test`| Run pytest                           |
| `make fmt` | Auto-format with Black + Ruff        |
| `make lint`| Lint with Ruff                       |

## PySpark MLlib path (parallel to scikit-learn)

A second, independent training path reimplements preprocessing + models with
Spark DataFrame + MLlib APIs, reusing the **same** processed parquet splits and
schema column lists so the two are directly comparable.

```bash
# Train all MLlib models → logs to the "churn-spark" MLflow experiment
python pipelines/train_spark.py

# Train a single estimator
python pipelines/train_spark.py --model gbt

# Head-to-head Spark-vs-sklearn benchmark (writes artifacts/reports/*.md + *.csv + charts)
python pipelines/benchmark_spark.py
```

- Runs in **local mode only** (`SparkSession` with `master local[*]`) — no cluster.
- MLlib estimators: `LogisticRegression`, `RandomForestClassifier`, and
  **GBT (gradient-boosted trees)** — MLlib's stand-in for XGBoost (MLlib has no
  native XGBoost). True XGBoost-on-Spark needs the external `xgboost` PySpark
  integration (`xgboost.spark.SparkXGBClassifier`), left optional here.
- Spark runs log to a distinct experiment (`churn-spark`) and register under a
  separate model name (`churn-classifier-spark`) so they never collide with the
  sklearn path.

> **Note on the benchmark.** On this tiny (~7k-row) dataset, Spark's JVM +
> scheduling overhead usually makes MLlib *slower* than scikit-learn. That is
> expected — the Spark path is about scaling to data that doesn't fit on one
> machine, not winning on a toy dataset.

### JDK requirement (Spark needs Java)

Spark requires **Java 8, 11, or 17**. Check what you have:

```bash
java -version
```

If Java is missing on macOS, install a supported JDK (17 recommended) with Homebrew:

```bash
brew install openjdk@17
# then, if java isn't picked up automatically, expose JAVA_HOME:
export JAVA_HOME="$(/usr/libexec/java_home -v 17)"
```

> **Python 3.12+ note.** PySpark 3.5.x still imports the stdlib `distutils`,
> which was removed in Python 3.12. If you hit `ModuleNotFoundError: No module
> named 'distutils'`, install the compatibility shim: `pip install "setuptools<81"`.

## Configuration

Non-secret settings (topic names, ports, paths) live in `configs/app.yaml`.  
Secrets (passwords, keys) are read from `.env` — copy `.env.example` to get started.

```python
from churnops.config import settings

print(settings.kafka_bootstrap_servers)
print(settings.mlflow_tracking_uri)
```
