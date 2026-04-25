.PHONY: setup db-up db-down db-migrate data-fetch features train eval paper test lint format full-report sync sync-data

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

# ── Remote sync ────────────────────────────────────────────────────────────────
# SERVER / REMOTE_DIR can be overridden on the command line:
#   make sync SERVER=user@host REMOTE_DIR=/path/on/server
SERVER     ?= jash@10.21.186.205
REMOTE_DIR ?= /home/jash/trading-agent/Trading_Bot

# Syncs source code, configs, scripts, docker, tests — nothing that is
# generated locally (venv, caches, downloaded data, model artefacts).
sync:
	rsync -azvh --progress --checksum \
		--exclude='.venv/' \
		--exclude='.git/' \
		--exclude='__pycache__/' \
		--exclude='*.pyc' \
		--exclude='.mypy_cache/' \
		--exclude='.ruff_cache/' \
		--exclude='.pytest_cache/' \
		--exclude='*.egg-info/' \
		--exclude='.hydra/' \
		--exclude='outputs/' \
		--exclude='mlruns/' \
		--exclude='checkpoints/' \
		--exclude='data/' \
		./* $(SERVER):$(REMOTE_DIR)

# Like sync, but also transfers the data/ directory (panels + raw cache).
# Use once after build_features.py if you don't want to re-download on server.
sync-data:
	rsync -azvh --progress --checksum \
		--exclude='.venv/' \
		--exclude='.git/' \
		--exclude='__pycache__/' \
		--exclude='*.pyc' \
		--exclude='.mypy_cache/' \
		--exclude='.ruff_cache/' \
		--exclude='.pytest_cache/' \
		--exclude='*.egg-info/' \
		--exclude='.hydra/' \
		--exclude='outputs/' \
		--exclude='mlruns/' \
		--exclude='checkpoints/' \
		./ $(SERVER):$(REMOTE_DIR)
