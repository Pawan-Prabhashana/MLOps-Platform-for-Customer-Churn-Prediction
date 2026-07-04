# ruff: noqa: E402
"""Local-dev launcher: start the FastAPI churn API with uvicorn on :8000.

Usage
-----
    python pipelines/serve_api.py
    python pipelines/serve_api.py --host 127.0.0.1 --port 8001
    python pipelines/serve_api.py --reload          # hot-reload for development
    python pipelines/serve_api.py --model-source joblib  # skip MLflow registry
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import yaml


def _load_api_config() -> dict:
    with (_REPO_ROOT / "configs" / "api.yaml").open() as f:
        return yaml.safe_load(f)


def parse_args(argv=None):
    cfg = _load_api_config().get("server", {})
    p = argparse.ArgumentParser(description="Start the ChurnOps FastAPI prediction server.")
    p.add_argument("--host", default=cfg.get("host", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(cfg.get("port", 8000)))
    p.add_argument("--reload", action="store_true", default=bool(cfg.get("reload", False)),
                   help="Enable uvicorn hot-reload (dev only).")
    p.add_argument("--log-level", default=cfg.get("log_level", "info"))
    p.add_argument("--model-source", default=None, choices=["registry", "joblib"],
                   help="Override the model source (sets MODEL_SOURCE env var).")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    if args.model_source:
        os.environ["MODEL_SOURCE"] = args.model_source

    import uvicorn

    print(f"\nStarting ChurnOps API on http://{args.host}:{args.port}")
    print(f"  Docs      : http://localhost:{args.port}/docs")
    print(f"  Health    : http://localhost:{args.port}/health")
    print(f"  Predict   : POST http://localhost:{args.port}/predict")
    print(f"  Model src : {os.environ.get('MODEL_SOURCE', 'from configs/api.yaml')}\n")

    uvicorn.run(
        "churnops.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
