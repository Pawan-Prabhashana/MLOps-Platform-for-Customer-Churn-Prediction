.PHONY: up down logs ps install test fmt lint

# ── Docker Compose ────────────────────────────────────────────────────────────

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

ps:
	docker compose ps

# ── Python ────────────────────────────────────────────────────────────────────

install:
	pip install -e ".[dev]"

test:
	pytest

fmt:
	black src tests
	ruff check --fix src tests

lint:
	ruff check src tests
