#!/usr/bin/env bash
# Host (macOS, Apple GPU) wrapper for the P3 refit. Run nightly by launchd
# (ops/launchd/com.paperbook.refit-adaptive.plist); refits only when due.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
set -a; [ -f .env ] && . ./.env; set +a
mkdir -p logs/retrain
exec >> "logs/retrain/refit_host_$(date +%F).log" 2>&1
echo "=== $(date '+%F %T %Z') refit_adaptive --if-due ==="
uv run python scripts/refit_adaptive.py --if-due
echo "exit=$?"
