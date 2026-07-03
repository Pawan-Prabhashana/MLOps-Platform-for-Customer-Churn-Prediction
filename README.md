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

## Configuration

Non-secret settings (topic names, ports, paths) live in `configs/app.yaml`.  
Secrets (passwords, keys) are read from `.env` — copy `.env.example` to get started.

```python
from churnops.config import settings

print(settings.kafka_bootstrap_servers)
print(settings.mlflow_tracking_uri)
```
