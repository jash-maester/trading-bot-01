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

# ── Windows training box (RTX 4060) ────────────────────────────────────────────
.PHONY: win-check win-push win-push-data win-push-env win-setup win-test win-shell win-gpu \
        win-services win-services-down

## Connectivity, path, docker and GPU in one call.
win-check:
	@$(SSH) $(SERVER) 'hostname; "---"; \
	  if (Test-Path "D:/trading-bot-01") { "path OK" } else { "PATH MISSING" }; \
	  docker info --format "docker {{.ServerVersion}} os={{.OSType}} running={{.ContainersRunning}}"; \
	  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader'

## Code, configs, scripts, tests, docs. Never data, never secrets.
win-push:
	$(WINRSYNC) $(SYNC_EXCLUDES) --exclude='/data/' ./ $(SERVER):$(REMOTE_DIR)/

## The panels and OHLCV store (~342 MB). Slow first time, incremental after.
win-push-data:
	$(WINRSYNC) $(SYNC_EXCLUDES) ./data/ $(SERVER):$(REMOTE_DIR)/data/

## Push .env to the training box. Deliberately a separate target, not part of
## win-push: secrets should move on an explicit command, never as a side effect
## of syncing code. Re-run it whenever the Kite access token is refreshed — it
## expires around 6am daily, so a stale copy will fail auth mid-run.
win-push-env:
	@rsync -az --rsync-path="$(RSYNC_PATH)" -e "$(SSH)" .env $(SERVER):$(REMOTE_DIR)/.env
	@$(SSH) $(SERVER) 'wsl.exe -e bash -lc "chmod 600 $(REMOTE_DIR)/.env && ls -l $(REMOTE_DIR)/.env"'

## Bring up Postgres + MLflow on the training box via docker compose. Unlike the
## Mac mini, that box HAS Docker, so the compose path works there unmodified.
## Bring up Postgres + MLflow on the training box. Unlike the Mac mini, that box
## HAS Docker, so the compose path works there unmodified. The logic lives in
## scripts/win_bootstrap.sh because PowerShell mangles inline pipes and loops,
## and because the box's ~/.docker/config.json uses "desktop.exe" as its
## credential helper — that needs an interactive Windows logon and fails over SSH
## with "A specified logon session does not exist", even for public images that
## need no credentials. The script points DOCKER_CONFIG at a project-local
## config to sidestep it without touching theirs.
win-services:
	@$(SSH) $(SERVER) 'wsl.exe -e bash $(REMOTE_DIR)/scripts/win_bootstrap.sh services'

win-services-down:
	@$(SSH) $(SERVER) 'wsl.exe -e bash -lc "cd $(REMOTE_DIR) && DOCKER_CONFIG=$(REMOTE_DIR)/.docker docker compose -f docker/docker-compose.yml down"'

## One-time remote bootstrap: uv, venv, deps, CUDA probe. Run after win-push.
## Logic lives in scripts/win_bootstrap.sh — see the comment there for why it is
## a file and not an inline command (PowerShell re-parses argv in transit).
win-setup:
	@$(SSH) $(SERVER) 'wsl.exe -e bash $(REMOTE_DIR)/scripts/win_bootstrap.sh setup'

## Full check suite on the training box.
win-test:
	@$(SSH) $(SERVER) 'wsl.exe -e bash $(REMOTE_DIR)/scripts/win_bootstrap.sh test'

win-gpu:
	@$(SSH) $(SERVER) 'nvidia-smi'

## Interactive WSL shell in the project directory.
win-shell:
	@$(SSH) -t $(SERVER) 'wsl.exe -e bash -lc "cd $(REMOTE_DIR) && exec bash -l"'

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
# Training box: Windows 11 + RTX 4060, reached over OpenSSH. Its default shell is
# PowerShell, which mangles rsync's server-side argv — so rsync is invoked through
# WSL via --rsync-path. Note `wsl.exe`, not bare `wsl`: PowerShell resolves the
# former and fails on the latter. The path is the WSL view of D:\trading-bot-01.
SERVER     ?= jashm@jashtuf.tail6c36a.ts.net
REMOTE_DIR ?= /mnt/d/trading-bot-01
RSYNC_PATH ?= wsl.exe rsync
SSH        ?= ssh -o BatchMode=yes
RSH         = $(SSH)
WINRSYNC    = rsync -azvh --progress --rsync-path="$(RSYNC_PATH)" -e "$(SSH)"

# Generated locally, never shipped: caches, venv, artefacts, secrets.
# NOTE the leading slash on '/data/' below and in win-push: an unanchored
# 'data/' matches at EVERY level, which silently excluded src/trader/data/
# and left the training box with a `trader` package missing its data module.
SYNC_EXCLUDES = \
	--exclude='.venv/' --exclude='.git/' --exclude='__pycache__/' --exclude='*.pyc' \
	--exclude='.mypy_cache/' --exclude='.ruff_cache/' --exclude='.pytest_cache/' \
	--exclude='*.egg-info/' --exclude='.hydra/' --exclude='outputs/' \
	--exclude='mlruns/' --exclude='checkpoints/' --exclude='logs/' \
	--exclude='.env' --exclude='*.bak' --exclude='reports/*.html' \
	--exclude='/backups/'

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
		--exclude='/data/' \
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
