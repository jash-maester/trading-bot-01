.PHONY: setup db-up db-down db-migrate data-fetch features train eval paper test lint format full-report

COMPOSE = docker compose -f docker/docker-compose.yml

setup:
	uv sync --all-extras

db-up:
	$(COMPOSE) up -d postgres mlflow

db-down:
	$(COMPOSE) down

db-migrate:
	uv run alembic upgrade head

data-fetch:
	uv run python scripts/fetch_data.py $(ARGS)

features:
	uv run python scripts/build_features.py $(ARGS)

train:
	uv run python scripts/train.py $(ARGS)

eval:
	uv run python scripts/evaluate.py $(ARGS)

paper:
	uv run python scripts/paper_run.py $(ARGS)

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

full-report:
	uv run python scripts/full_report.py $(ARGS)
