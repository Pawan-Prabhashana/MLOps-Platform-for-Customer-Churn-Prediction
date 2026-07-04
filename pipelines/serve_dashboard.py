# ruff: noqa: E402
"""serve_dashboard.py — Local-dev launcher: start the Streamlit business dashboard on :8501.

Thin wrapper around `streamlit run dashboard/app.py`. Streamlit doesn't expose
a clean in-process run() call (unlike uvicorn), so this shells out to the
streamlit CLI in a subprocess — this file only resolves the port/host from
configs/dashboard.yaml and builds the command.

Usage
-----
    python pipelines/serve_dashboard.py
    python pipelines/serve_dashboard.py --port 8502
    streamlit run dashboard/app.py --server.port 8501   # equivalent, direct
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import yaml


def _load_dashboard_config() -> dict:
    with (_REPO_ROOT / "configs" / "dashboard.yaml").open() as f:
        return yaml.safe_load(f)


def parse_args(argv=None):
    cfg = _load_dashboard_config().get("server", {})
    p = argparse.ArgumentParser(description="Start the ChurnOps business dashboard (Streamlit).")
    p.add_argument("--port", type=int, default=int(cfg.get("port", 8501)))
    p.add_argument("--host", default="0.0.0.0")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    app_path = _REPO_ROOT / "dashboard" / "app.py"

    print(f"\nStarting ChurnOps business dashboard on http://localhost:{args.port}")
    print(f"  App file : {app_path}")
    print("  Stop with Ctrl-C\n")

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", str(args.port),
        "--server.address", args.host,
    ]
    subprocess.run(cmd, check=False, cwd=str(_REPO_ROOT))


if __name__ == "__main__":
    main()
