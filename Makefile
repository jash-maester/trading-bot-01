.PHONY: setup db-up db-down db-migrate data-fetch features train eval paper test lint format full-report sync sync-data \
        services-up services-down services-status mlflow-up mlflow-down pg-up pg-down

COMPOSE = docker compose -f docker/docker-compose.yml
PG_BIN  = /opt/homebrew/opt/postgresql@16/bin
MLFLOW_PORT ?= 5555

setup:
	uv sync --all-extras

# ── Docker path (CUDA box / any machine with Docker) ──────────────────────────
# NOTE: there is no Docker on the Mac mini. Use `services-up` there instead.
db-up:
	$(COMPOSE) up -d postgres mlflow

db-down:
	$(COMPOSE) down

db-migrate:
	uv run alembic upgrade head

# ── Native path (Mac mini — no Docker) ────────────────────────────────────────
# Postgres 16 via Homebrew services, MLflow from the project venv.
# See HANDOFF.md §5.3.
pg-up:
	brew services start postgresql@16
	@for i in $$(seq 1 30); do $(PG_BIN)/pg_isready -h localhost -p 5432 -q && break; sleep 1; done
	@$(PG_BIN)/pg_isready -h localhost -p 5432

pg-down:
	brew services stop postgresql@16

mlflow-up:
	@mkdir -p mlruns logs
	@if curl -sf http://127.0.0.1:$(MLFLOW_PORT)/health >/dev/null 2>&1; then \
		echo "MLflow already up on $(MLFLOW_PORT)"; \
	else \
		nohup .venv/bin/mlflow server --host 127.0.0.1 --port $(MLFLOW_PORT) \
			--backend-store-uri "sqlite:///$(CURDIR)/mlruns/mlflow.db" \
			--artifacts-destination "$(CURDIR)/mlruns/artifacts" \
			--serve-artifacts > logs/mlflow.log 2>&1 & \
		for i in $$(seq 1 60); do curl -sf http://127.0.0.1:$(MLFLOW_PORT)/health >/dev/null 2>&1 && break; sleep 1; done; \
		curl -sf http://127.0.0.1:$(MLFLOW_PORT)/health >/dev/null && echo "MLflow up on $(MLFLOW_PORT)" || \
			{ echo "MLflow failed to start — see logs/mlflow.log"; tail -20 logs/mlflow.log; exit 1; }; \
	fi

mlflow-down:
	@pkill -f "mlflow server --host 127.0.0.1 --port $(MLFLOW_PORT)" && echo "MLflow stopped" || echo "MLflow not running"

services-up: pg-up mlflow-up
	@echo "Postgres 5432 + MLflow $(MLFLOW_PORT) up. Run 'make db-migrate' once on a fresh DB."

services-down: mlflow-down pg-down

services-status:
	@$(PG_BIN)/pg_isready -h localhost -p 5432 || true
	@printf 'mlflow: '; curl -sf http://127.0.0.1:$(MLFLOW_PORT)/health || echo "DOWN"
	@echo

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
		./ $(SERVER):$(REMOTE_DIR)

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
